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
