#!/usr/bin/env python
# coding: utf-8

# ## Mini example for scikit-survival estimators

import numpy as np
from survwrap import load_test_data

test_X, test_y = load_test_data()


def test_sksurv_data_is_loading(X=test_X):
    assert X.shape == (198, 84)


def test_sksurv_coxnet(X=test_X, y=test_y):
    # Fit penalized Cox model (from scikit-survival)
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import cross_val_score

    model = CoxnetSurvivalAnalysis()
    model.fit(X, y)
    pred = model.predict(X)

    # Standard functions from scikit-learn can be used with scikit-survival models

    cv_score = cross_val_score(CoxnetSurvivalAnalysis(), X, y)
    # testing assertion: arrays uguali alla 3 decimale

    np.testing.assert_array_almost_equal(
        cv_score, [0.617, 0.532, 0.589, 0.605, 0.752], decimal=3
    )


def test_wrapped_coxnet(X=test_X, y=test_y):
    # Fit penalized Cox model (from scikit-survival)
    import survwrap as sw

    # from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import cross_val_score

    model = sw.CoxNet(verbose=True)
    model.fit(X, y)
    pred = model.predict(X)
    fit_score = model.score(X, y)

    # assert on simple score
    assert fit_score.round(3) == 0.946

    # assert on 3-fold cross-validation score
    cv_score = cross_val_score(model, X, y, cv=3)
    # testing assertion: arrays uguali alla 3° decimale
    np.testing.assert_array_almost_equal(cv_score, [0.687, 0.592, 0.598], decimal=3)


### ------------

# def stub() :
# ## Scikit-learn compatibility
# Scikit-learn has a checker for estimators to see if they conform to their specification.

# # In[8]:

# from sklearn.utils.estimator_checks import check_estimator


# # Scikit-survival models do not necessarily pass ;-)

# # In[9]:


# try:
#     check_estimator(CoxnetSurvivalAnalysis())
# except Exception as e:
#     print(e.__class__.__name__)
#     print(e)
