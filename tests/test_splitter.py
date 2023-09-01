"tests on bechmark datasets integrity and format conversions"

import survwrap as tosa

_shapes = [(138, 84), (60, 84), (138,), (60,)]
_events = [51, 36, 15]


def test_stratified_splitter():
    "test util.survival_train_test_split function"
    X, y = tosa.load_test_data()
    splits = tosa.survival_train_test_split(X, y, rng_seed=2308, test_size=0.3)
    spl_shapes = [_.shape for _ in splits]
    assert spl_shapes == _shapes
    events = [tosa.get_indicator(_).sum() for _ in [y, splits[2], splits[3]]]
    assert events == _events
