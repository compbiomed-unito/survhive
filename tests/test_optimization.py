"tests on bechmark datasets integrity and format conversions"

import survwrap as tosa


X, y = tosa.load_test_data()


def test_topology_grid():
    "test util.generate_topology_grid function"
    _topologies = [[3], [4], [5], [7], [9], [3, 3], [4, 4], [5, 5], [7, 7], [9, 9]]

    assert tosa.generate_topology_grid(9, max_layers=2) == _topologies


def test_guess_tries():
    "test optimization._guess_tries function"
    dhs_guess = tosa.optimization._guess_tries(
        tosa.DeepHitSingle.get_parameter_grid(max_width=X.shape[1]), fraction=0.2
    )
    assert dhs_guess == 16
    dsm = tosa.optimization._guess_tries(
        tosa.DeepSurvivalMachines.get_parameter_grid(max_width=X.shape[1]),
    )
    assert dsm == 24


def test_grid_opt_coxnet():
    "test grid optimization on CoxNet"

    estimator = tosa.CoxNet(rng_seed=2309)

    test_grid = estimator.get_parameter_grid()
    del test_grid["l1_ratio"]
    grid_coxnet, grid_coxnet_params, grid_coxnet_search = tosa.optimize(
        estimator,
        X,
        y,
        mode="sklearn-grid",
        user_grid=test_grid,
        cv=tosa.survival_crossval_splitter(X, y, n_splits=3, n_repeats=1),
        n_jobs=2,
    )
    assert grid_coxnet.score(X, y).round(3) == 0.946


def test_rs_opt_coxnet():
    "test random-search optimization on CoxNet"

    estimator = tosa.CoxNet(rng_seed=2309)
    rs_coxnet, rs_coxnet_params, rs_coxnet_search = tosa.optimize(
        estimator,
        X,
        y,
        mode="sklearn-random",
        tries=10,
        cv=tosa.survival_crossval_splitter(X, y, n_splits=3, n_repeats=1),
        n_jobs=2,
    )
    assert rs_coxnet.score(X, y).round(3) == 0.931
