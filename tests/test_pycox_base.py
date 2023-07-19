#!/usr/bin/env python
# coding: utf-8

# ## Mini example for scikit-survival estimators

import numpy as np
from survwrap import load_test_data

test_X, test_y = load_test_data()


def test_wrapped_DHSingle(X=test_X, y=test_y):
    # Fit DeepHitSingle model (from pycox)
    import survwrap as tosa

    # from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import cross_val_score

    model = tosa.DeepHitSingle(
        epochs=100,
        #      layer_sizes=[10, 10],
        learning_rate=0.05,
        batch_size=len(y) // 2,
        device="cpu",
        rng_seed=2307,
    )
    model.fit(X, y)
    pred = model.predict(X)
    fit_score = model.score(X, y)

    # assert on simple score
    assert fit_score.round(2) == 0.96

    # assert on 3-fold cross-validation score
    cv_score = cross_val_score(model, X, y, cv=3)
    # testing assertion: arrays uguali alla 3° decimale
    np.testing.assert_array_almost_equal(cv_score, [0.512, 0.475, 0.692], decimal=3)


### ------------
