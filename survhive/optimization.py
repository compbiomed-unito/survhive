from math import prod
from functools import reduce
from pandas import DataFrame, Index
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from .util import survival_crossval_splitter


## hyperparameter optimization


def generate_topology_grid(max_width, max_layers=3):
    "return a list of net topologies to be used in hyperparameter optimization"
    from math import log

    start = 3
    base = 1.3
    topologies = []
    mono = [
        round(start * base ** (_))
        for _ in range(max_width)
        if _ < int((log(max_width) - log(start)) / log(base)) + 1
    ]
    for n_layers in range(1, max_layers + 1):
        topologies.extend([[_] * n_layers for _ in mono])
    return topologies


def get_grid_size(grid):
    "calculate the number of points in a grid"

    return prod([len(_) for _ in grid.values()])


def _guess_tries(grid, fraction=0.05):
    "calculate the number of points in a grid and return fraction*points"

    return 1 + int(fraction * get_grid_size(grid))


def optimize(
    estimator,
    X,
    y,
    mode="sklearn-grid",
    user_grid=[],
    cv=None,
    scoring=None,
    tries=None,
    n_jobs=1,
    refit=True,
):
    "hyperparameter optimization of estimator"

    if cv is None:
        cv = survival_crossval_splitter(X, y, rng_seed=estimator.rng_seed)
    if not user_grid:
        user_grid = estimator.get_parameter_grid(max_width=X.shape[1])

    sk_common_params = dict(
        estimator=estimator, refit=refit, cv=cv, scoring=scoring, n_jobs=n_jobs
    )
    if mode == "sklearn-grid":
        gs = GridSearchCV(
            param_grid=user_grid,
            **sk_common_params,
        )
    elif mode == "sklearn-random":
        if not tries:
            tries = _guess_tries(user_grid)
        print("Random search tries:", tries)
        gs = RandomizedSearchCV(
            param_distributions=user_grid,
            random_state=estimator.rng_seed,
            n_iter=tries,
            **sk_common_params,
        )
    else:
        raise ValueError(f'unknown mode parameter: "{mode}"')
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs


def get_model_scores_df(search):
    """
    Returns a pandas dataframe containing rank, avg_cv_score, std_cv_score,
    params for each score specified in an optimization search result.
    """

    if search.scoring:
        # multiple scoring
        zcorez = search.scoring.keys()
        zcored_by = search.refit
    else:
        zcorez = ["score"]
        zcored_by = "score"

    labelz = [
        "_test_".join([_f, _z]) for _z in zcorez for _f in ["rank", "mean", "std"]
    ] + ["params", "mean_fit_time", "std_fit_time"]

    return DataFrame(
        [search.cv_results_[_] for _ in labelz], index=labelz
    ).T.sort_values("_".join(["rank_test", zcored_by]))


def get_model_top_ranking_df(search):
    """
    Returns a pandas dataframe containing the top-ranking solutions
    for each score specified in an optimization search result.
    This is a subset of what reported from the get_model_scores_df function.
    """
    _scores_df = get_model_scores_df(search)
    # top_scorers
    ranks = [_ for _ in list(_scores_df.columns) if _.startswith("rank")]
    _top_scorers_ndx = reduce(
        Index.append, [_scores_df.index[_scores_df[_] == 1] for _ in ranks]
    ).unique()
    return _scores_df.loc[_top_scorers_ndx]
