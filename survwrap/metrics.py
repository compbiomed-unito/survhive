"""Time-dependent score function have the signature (y_true, y_pred, times)"""

import numpy
import sksurv
from sklearn.metrics import roc_auc_score, brier_score_loss
from .util import get_indicator, get_time


def concordance_index_score(y_true, y_pred, return_all=False):
    # standardized calling signature of scikit-survival concordance_index_censored
    r = sksurv.metrics.concordance_index_censored(
        get_indicator(y_true), get_time(y_true), y_pred
    )
    return r if return_all else r[0]


def _estimate_concordance_index_antolini(
    event_indicator, event_time, estimate, weights, tied_tol=1e-8
):
    # taken from sksurv.metrics, generalized to a time-dependant estimate matrix
    # not meant to be called directly but through the concordance_index_td_scorer,
    # since it only works if estimate has the y times as its second index

    order = numpy.argsort(event_time)

    tied_time = None

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask, tied_time in sksurv.metrics._iter_comparable(
        event_indicator, event_time, order
    ):
        est_i = estimate[order[ind], order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask], order[ind]]

        assert (
            event_i
        ), f"got censored sample at index {order[ind]}, but expected uncensored"

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    if tied_time is None:
        raise sksurv.exceptions.NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_antolini_scorer(estimator, X, y, return_all=False):
    """Naive extension of concordance index to time-dependent predictions."""
    r = _estimate_concordance_index_antolini(
        event_indicator=get_indicator(y),
        event_time=get_time(y),
        estimate=1.0 - estimator.predict_survival(X, get_time(y)),  # use failure
        weights=numpy.full(len(y), 1.0),
    )
    return r if return_all else r[0]


def make_survival_scorer(
    score_func,
    needs="failure",
    classification=False,
    aggregate="mean",
    time_mode="events",
    time_values=None,
    **kwargs,
):
    """
    Create a time-dependent survival scoring function for survival analysis.

    Parameters:
    - score_func (callable, with signature (y_pred, y_true)): A function that
      computes a score based on predicted and true values.
    - needs (str, optional): The type of predictions needed. Either "failure"
      or "survival" probability predictions. Default is "failure".
    - classification (bool, optional): If True, treat score_func as a
      classification score and run it on positive/negative events computed
      separately for each time point. Default is False.
    - aggregate (str, optional): The method to aggregate scores over different
      time points. Options include 'mean', 'median', 'sum', or 'no' for no
      aggregation. Default is 'mean'.
    - time_mode (str, optional): The mode for specifying prediction times.
      Options are "events" (using event times), "quantiles" (using quantiles of
      event times), or "absolute" (using specified absolute time values).
      Default is "events".
    - time_values (array-like or float, optional): The time values depending on
      the chosen time_mode. If time_mode is "events", time_values should be
      None. If time_mode is "quantiles", time_values should be an array of
      quantiles between 0 and 1. If time_mode is "absolute", time_values should
      be an array-like object or a float representing absolute time values.
    - **kwargs: Additional keyword arguments to be passed to the underlying
      score_func.

    Returns:
    - scorer (callable with signature (estimator, X, y)): A time-dependent
      scoring function that computes score_func at different time points and
      aggregate the results.

    Notes:
    - The resulting scorer can be used as a standard scikit-learn scorer with
      survival outcomes and survwrap models. See the example

    ```
    from survwrap import CoxNet, load_test_data
    from sklearn.metrics import roc_auc_score, brier_score_loss
    roc_auc_at_quartiles = make_survival_scorer(roc_auc_score, classification=True,
                                                time_mode='quantiles',
                                                time_values=[0.25, 0.5, 0.75])
    brier_at_quartiles = make_survival_scorer(lambda *args: -brier_score_loss(*args),
                                              classification=True,
                                              time_mode='quantiles',
                                              time_values=[0.25, 0.5, 0.75]),

    X, y = load_test_data('veterans_lung_cancer')

    cross_val_score(CoxNet(), X, y, scoring=roc_auc_at_quartiles)
    ```
    """

    def scorer(estimator, X, y):
        event_times = get_time(y)[get_indicator(y)]

        # get evaluation times
        if time_mode == "events":
            pred_times = event_times
        elif time_mode == "quantiles":
            pred_times = numpy.quantile(event_times, time_values)
        elif time_mode == "absolute":
            pred_times = time_values
        else:  # keep as is, must be a scalar or sequence
            raise ValueError('needs must be either "events", "quantiles" or "absolute"')

        # compute predictions at pred_times
        if needs == "failure":
            y_pred = 1.0 - estimator.predict_survival(X, pred_times)
        elif needs == "survival":
            y_pred = estimator.predict_survival(X, pred_times)
        else:
            raise ValueError('needs must be either "failure" or "survival"')

        # run score_func at each time
        scores = []
        for p, t in zip(y_pred.T, pred_times):
            if classification:
                y_time = get_time(y)
                y_ind = get_indicator(y)

                informative = (y_time > t) | y_ind
                positive = (y_time <= t) & y_ind

                score = score_func(positive[informative], p[informative])
            else:
                score = score_func(y, p)
            if score != score:
                print(f"bad survival score at time {t} computed by {score_func}")
            scores.append(score)

        # aggregate scores for different times
        if aggregate == "no":
            return numpy.array(scores)
        else:
            if hasattr(numpy, aggregate):
                return getattr(numpy, aggregate)(scores)
            else:
                raise ValueError(f"unknonw aggregate value `{aggregate}`")

    scorer.__name__ = score_func.__name__ + "_td_scorer"

    return scorer


_qt = dict(time_mode="quantiles", time_values=numpy.linspace(0, 1, 5)[1:-1])
_SCORERS = {
    "c-index-antolini": concordance_index_antolini_scorer,
    "c-index-quartiles": make_survival_scorer(
        concordance_index_score, classification=False, **_qt
    ),  # FIXME this is not a good score, maybe remove it from this list
    "roc-auc-quartiles": make_survival_scorer(
        roc_auc_score, classification=True, **_qt
    ),
    "neg-brier-quartiles": make_survival_scorer(
        lambda *args: -brier_score_loss(*args), classification=True, **_qt
    ),
}
