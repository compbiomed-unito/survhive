"tests on bechmark datasets integrity and format conversions"

import survwrap as tosa

_shapes = [(138, 84), (60, 84), (138,), (60,)]
_events = [51, 36, 15]
_cv_events = [17] * 6

X, y = tosa.load_test_data()


def test_stratified_splitter():
    "test util.survival_train_test_split function"
    splits = tosa.survival_train_test_split(X, y, rng_seed=2308, test_size=0.3)
    spl_shapes = [_.shape for _ in splits]
    assert spl_shapes == _shapes
    events = [tosa.get_indicator(_).sum() for _ in [y, splits[2], splits[3]]]
    assert events == _events


def test_stratified_cv_splitter():
    "test util.survival_crossval_splitter function"
    cv_splitter = tosa.survival_crossval_splitter(
        X, y, n_splits=3, n_repeats=2, rng_seed=2309
    )
    cv_events = [tosa.get_indicator(y[_[1]]).sum() for _ in cv_splitter]
    assert cv_events == _cv_events
