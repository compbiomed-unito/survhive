#!/usr/bin/env python
# coding: utf-8

import numpy

# ## Mini example for scikit-survival estimators


def load_toy_data():
    "Load standard dataset for testing"
    import sksurv.datasets
    from sklearn.preprocessing import OneHotEncoder

    X, y = sksurv.datasets.load_breast_cancer()
    X = numpy.concatenate(
        [
            X.select_dtypes("float"),
            OneHotEncoder(sparse_output=False).fit_transform(
                X.select_dtypes("category")
            ),
        ],
        axis=1,
    )
    return X, y

test_X, test_y = load_toy_data()

def test_sksurv_data_is_loading(X=test_X):
    assert X.shape == (198, 84)



# Fit penalized Cox model (from scikit-survival)
# def stub() :
def test_sksurv_coxnet(X=test_X, y=test_y) :
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    model = CoxnetSurvivalAnalysis()
    model.fit(X, y)
    pred = model.predict(X)


# # Standard functions from scikit-learn can be used with scikit-survival models

# # In[6]:


# from sklearn.model_selection import cross_val_score


# # In[7]:


# cross_val_score(CoxnetSurvivalAnalysis(), X, y)

### ------------

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
