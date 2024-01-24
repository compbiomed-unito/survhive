import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array

from .metrics import concordance_index_antolini_scorer


@dataclass
class SurvivalEstimator(BaseEstimator):
    """
    This is a minimal (empty) estimator that passes the sk-learn check_estimator tests.

    Dataclasses can be useful to avoid long init functions and it appears to work.
    - BaseEstimator include the get/set_params methods that are required.
    - check_X_y and check_array implement checks (required by check_estimator function)
        on the input data.
    """

    package = None
    model = None
    rng_seed: int = None

    def _seed_rngs(self):
        "Seed the random number generators involved in the model fit"
        pass

    def fit(self, X, y):
        "fit the model"
        X, y = check_X_y(X, y)
        return self

    def predict(self, X):
        "do a prediction using a fit model"
        X = check_array(X)
        return np.full(shape=X.shape[0], fill_value=(1,))

    def predict_survival(self, X, times):
        X = check_array(X)
        return np.full(shape=(X.shape[0], len(times)), fill_value=(1,))

    # def harrell_score(self, y_true, y_pred, *args, **kwargs):
    #     "return Harrell's C-index for a prediction"

    #     return concordance_index_censored(
    #         event_indicator=get_indicator(y_true),
    #         event_time=get_time(y_true),
    #         estimate=y_pred,
    #         *args,
    #         **kwargs,
    #     )

    def score(self, X, y):
        "return the Harrell's c-index as a sklearn score"
        X, y = check_X_y(X, y)
        return concordance_index_antolini_scorer(self, X, y)
