"""Time-dependent score function have the signature (y_true, y_pred, times)"""

import numpy
import sksurv
from .util import get_indicator, get_time


def concordance_index_score(y_true, y_pred, return_all=False):
    # standardized calling signature of scikit-survival concordance_index_censored
    r = sksurv.metrics.concordance_index_censored(
        get_indicator(y_true), get_time(y_true), y_pred
    )
    return r if return_all else r[0]


def _estimate_concordance_index_td(
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


def concordance_index_td_scorer(estimator, X, y, return_all=False):
    """Naive extension of concordance index to time-dependent predictions."""
    r = _estimate_concordance_index_td(
        event_indicator=get_indicator(y),
        event_time=get_time(y),
        estimate=1.0 - estimator.predict_survival(X, get_time(y)),  # use failure
        weights=numpy.full(len(y), 1.0),
    )
    return r if return_all else r[0]


def brier_score(y_true, y_pred, times):
    # Time-dependent brier score
    # y_true: shape n_samples
    # y_pred: probability of event at time, shape n_samples x n_times
    # times: prediction times, shape n_times

    y_time = get_time(y_true)
    y_ind = get_indicator(y_true)
    if len(y_pred.shape) == 2:
        y_time = y_time.reshape((-1, 1))
        y_ind = y_ind.reshape((-1, 1))

    informative = (y_time > times) | y_ind
    positive = (y_time <= times) & y_ind
    err = y_pred[informative] - positive[informative]
    return numpy.square(err).mean()


def make_time_dependent_classification_score(score_func, aggregate="mean"):
    """Make a time-dependent survival metric from a classification metric.

    Given a time threshold t, a survival outcome can be reduced to a classification
    outcome by taking as positives all events occurring before t and as negatives
    all events or censorings occurring after t (censorings before t are discarded
    as non informative).  Using this reduction at multiple times we can extend any
    classification metric to survival.

    Parameters
    ----------
    score_func : callable
        Classification metric function with signature
        ``score_func(y_true, y_pred, **kwargs)

    """

    def td_score(y_true, y_pred, times, **kwargs):
        y_time = get_time(y_true)
        y_ind = get_indicator(y_true)
        if len(y_pred.shape) == 2:
            y_time = y_time.reshape((-1, 1))
            y_ind = y_ind.reshape((-1, 1))

        informative = (y_time > times) | y_ind
        positive = (y_time <= times) & y_ind

        if len(y_pred.shape) == 2:
            scores = numpy.array(
                [
                    score_func(positive[mask, i], y_pred[mask, i], **kwargs)
                    for i, mask in enumerate(informative.T)
                ]
            )
            if aggregate == "no":
                return scores
            elif aggregate == "mean":
                return numpy.mean(scores)
        else:
            return score_func(positive[informative], y_pred[informative], **kwargs)

    return td_score


def make_time_dependent_scorer(
    score_func, needs="failure", time_mode="events", time_values=None, **kwargs
):
    """Make a scorer from a time-dependent survival metric function.

    Parameters
    ----------
    score_func: callable
        Time-dependent score or loss function with signature
        ``score_func(y_true, y_pred, times, **kwargs)
    needs: string
        the type of survival prediction needed for td_score, one of
        `failure`: probability of event before a given time,
        `survival`: probability of event after a given time.
    times: string or sequence
        indicates at which times to evaluate td_score

    Return
    ------
    scorer: callable with signature (estimator, X, y)
    """

    def scorer(estimator, X, y):
        event_times = get_time(y)[get_indicator(y)]

        if time_mode == "events":
            pred_times = event_times
        elif time_mode == "quantiles":
            pred_times = numpy.quantile(event_times, time_values)
        elif time_mode == "absolute":
            pred_times = time_values
        else:  # keep as is, must be a scalar or sequence
            raise ValueError('needs must be either "events", "quantiles" or "absolute"')

        if needs == "failure":
            y_pred = 1.0 - estimator.predict_survival(X, pred_times)
        elif needs == "survival":
            y_pred = estimator.predict_survival(X, pred_times)
        else:
            raise ValueError('needs must be either "failure" or "survival"')
        return score_func(y, y_pred, pred_times, **kwargs)

    return scorer
