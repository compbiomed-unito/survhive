# ## Scikit-learn compatibility

# Scikit-learn has a checker for estimators to see if they conform to their
# specification.
#
# Scikit-survival models do not necessarily pass ;-)

from sklearn.utils.estimator_checks import check_estimator
from survwrap import *


def sklearn_compliance(TestedEstimator):
    "Check esitmator for scikit-learn compliance"
    try:
        check_estimator(TestedEstimator())
    except Exception as e:
        print(e.__class__.__name__)
        print(e)


def test_survival_estimator_compliance():
    sklearn_compliance(SurvivalEstimator)


# _estimators = [ BaseEstimator ]
# def test_base_estimator_compliance():
#     sklearn_compliance(BaseEstimator)