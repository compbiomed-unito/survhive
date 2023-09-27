from math import prod
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


def _guess_tries(grid, fraction=0.05):
    "calculate the number of points in a grid and return fraction*points"

    return 1 + int(fraction * prod([len(_) for _ in grid.values()]))


def optimize(
    estimator, X, y, mode="sklearn-grid", user_grid=[], cv=None, tries=None, n_jobs=1
):
    "hyperparameter optimization of estimator"
    if not cv:
        cv = survival_crossval_splitter(X, y, rng_seed=estimator.rng_seed)
    if not user_grid:
        user_grid = estimator.get_parameter_grid(max_width=X.shape[1])
    if mode == "sklearn-grid":
        gs = GridSearchCV(estimator, user_grid, refit=True, cv=cv, n_jobs=n_jobs)
    elif mode == "sklearn-random":
        if not tries:
            tries = _guess_tries(user_grid)
        print("Random search tries:", tries)
        gs = RandomizedSearchCV(
            estimator,
            user_grid,
            random_state=estimator.rng_seed,
            refit=True,
            cv=cv,
            n_iter=tries,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f'unknown mode parameter: "{mode}"')
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs


def get_top_models(search_results, top=10):
    """evaluate the top best scoring result from an hyperparameter optimization.
    Returns a tuple containing (rank, avg_cv_score, std_cv_score, params)
    """

    _cv_rez = search_results.cv_results_
    return sorted(
        zip(
            _cv_rez["rank_test_score"],
            _cv_rez["mean_test_score"],
            _cv_rez["std_test_score"],
            _cv_rez["params"],
        ),
        key=lambda x: x[0],
    )[:top]
