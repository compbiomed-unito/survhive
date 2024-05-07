"tests on bechmark datasets integrity and format conversions"

import survhive


X, y = survhive.load_test_data()


def test_topology_grid():
    "test util.generate_topology_grid function"
    _topologies = [[3], [4], [5], [7], [9], [3, 3], [4, 4], [5, 5], [7, 7], [9, 9]]

    assert survhive.generate_topology_grid(9, max_layers=2) == _topologies


def test_guess_tries():
    "test optimization._guess_tries function"
    dhs_guess = survhive.optimization._guess_tries(
        survhive.DeepHitSingle.get_parameter_grid(max_width=X.shape[1]), fraction=0.2
    )
    assert dhs_guess == 24
    dsm = survhive.optimization._guess_tries(
        survhive.DeepSurvivalMachines.get_parameter_grid(max_width=X.shape[1]),
    )
    assert dsm == 24


def test_grid_opt_coxnet():
    "test grid optimization on CoxNet"

    estimator = survhive.CoxNet(rng_seed=2403)

    test_grid = estimator.get_parameter_grid()
    del test_grid["l1_ratio"]
    grid_coxnet, grid_coxnet_params, grid_coxnet_search = survhive.optimize(
        estimator,
        X,
        y,
        mode="sklearn-grid",
        user_grid=test_grid,
        cv=survhive.survival_crossval_splitter(X, y, n_splits=3, n_repeats=1),
        n_jobs=2,
    )
    assert grid_coxnet.score(X, y).round(3) == 0.815
    # checking that get_model_top_scorers_df works for simple scoring
    assert survhive.get_model_top_ranking_df(grid_coxnet_search).shape == (1, 6)


def test_rs_opt_coxnet():
    "test random-search optimization on CoxNet"

    estimator = survhive.CoxNet(rng_seed=2309)
    rs_coxnet, rs_coxnet_params, rs_coxnet_search = survhive.optimize(
        estimator,
        X,
        y,
        mode="sklearn-random",
        tries=10,
        cv=survhive.survival_crossval_splitter(X, y, n_splits=3, n_repeats=1),
        n_jobs=2,
    )
    assert rs_coxnet.score(X, y).round(2) == 0.83


# def test_grid_opt_coxph():
#     "test grid optimization on coxph"

#     estimator = survhive.CoxPH(rng_seed=2311)

#     test_grid = estimator.get_parameter_grid()
#     test_grid["alpha"] = test_grid["alpha"][10:]
#     grid_coxph, grid_coxph_params, grid_coxph_search = survhive.optimize(
#         estimator,
#         X,
#         y,
#         mode="sklearn-grid",
#         user_grid=test_grid,
#         cv=survhive.survival_crossval_splitter(X, y, n_splits=3, n_repeats=1),
#         n_jobs=2,
#     )
#     assert grid_coxph.score(X, y).round(2) == 0.71
#     # also check that get_model_scores_df works for simple scoring
#     assert survhive.get_model_scores_df(grid_coxph_search).shape == (12, 6)
