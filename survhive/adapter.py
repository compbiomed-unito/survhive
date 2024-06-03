import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array

from .metrics import concordance_index_antolini_scorer

__all__ = [
    "SurvivalEstimator",
]


@dataclass
class SurvivalEstimator(BaseEstimator):
    """
    This is a minimal (empty) estimator that passes the sklearn check_estimator tests.

    SurvivalEstimator is implemented as a dataclass.
    """

    package = None
    """the package from which the frapped class is coming."""

    model = None

    rng_seed: int = None
    """random-number generator seed"""

    def _seed_rngs(self):
        "Seed the random number generators involved in the model fit"
        pass

    def fit(self, X, y):
        "fit the model"
        X, y = check_X_y(X, y)
        return self

    def predict(self, X):
        "do a (risk) prediction using a fit model"
        X = check_array(X)
        return np.full(shape=X.shape[0], fill_value=(1,))

    def predict_survival(self, X, times):
        "predict survival at given *times* using the fit model"

        X = check_array(X)
        return np.full(shape=(X.shape[0], len(times)), fill_value=(1,))

    def score(self, X, y):
        "return the Antolini average c-index as a sklearn score"
        X, y = check_X_y(X, y)
        return concordance_index_antolini_scorer(self, X, y)

    @staticmethod
    def get_parameter_grid(max_width: int):
        """set-up a default grid for parameter optimization.

        Arguments:
            max_width: the dimension of a neural-network layer

        Returns:
            a dictionary of parameters compatible with sklearn search methods
        """
        return dict()
