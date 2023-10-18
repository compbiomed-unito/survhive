"tests on bechmark datasets integrity and format conversions"

import survwrap
import numpy as np

X, y = survwrap.load_test_data()


def test_stratified_splitter():
    "test util.survival_train_test_split function"

    _shapes = [(138, 84), (60, 84), (138,), (60,)]
    _events = [51, 36, 15]

    splits = survwrap.survival_train_test_split(X, y, rng_seed=2308, test_size=0.3)
    spl_shapes = [_.shape for _ in splits]
    assert spl_shapes == _shapes
    events = [survwrap.get_indicator(_).sum() for _ in [y, splits[2], splits[3]]]
    assert events == _events


def test_stratified_cv_splitter():
    "test util.survival_crossval_splitter function"

    _cv_events = [17] * 6

    cv_splitter = survwrap.survival_crossval_splitter(
        X, y, n_splits=3, n_repeats=2, rng_seed=2309
    )
    cv_events = [survwrap.get_indicator(y[_[1]]).sum() for _ in cv_splitter]
    assert cv_events == _cv_events


def test_event_quantiles():
    "test util.event_quantiles function"
    time_quant = [726.5, 994.0, 1206.0, 1721.0, 2733.5]
    assert np.array_equal(
        survwrap.event_quantiles(y, quantiles=[0.25, 0.4, 0.5, 0.6, 0.75]), time_quant
    )
