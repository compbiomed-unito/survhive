# ## Scikit-learn compatibility

# Scikit-learn has a checker for estimators to see if they conform to their
# specification.

from sklearn.utils.estimator_checks import check_estimator
from survwrap import *


def sklearn_compliance(TestedEstimator):
    "Check esitmator for scikit-learn compliance"
    try:
        check_estimator(TestedEstimator())
    except Exception as e:
        print(e.__class__.__name__)
        print(e)


# Base dataclass


def test_survival_estimator_compliance():
    sklearn_compliance(SurvivalEstimator)


# CoxNet from scikit-survival


def test_wrapped_coxnet_compliance():
    sklearn_compliance(CoxNet)


# DeepHitSingle from pycox


def test_wrapped_deephitsingle_compliance():
    sklearn_compliance(DeepHitSingle)

# DeepSurvivalMachines from auton-survival


def test_wrapped_DSM_compliance():
    sklearn_compliance(DeepSurvivalMachines)

