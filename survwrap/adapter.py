from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array


@dataclass
class SurvivalEstimator(BaseEstimator):
    """
    This is a minimal (empty) estimator that passes the sk-learn check_estimator tests.

    Dataclasses can be useful to avoid long init functions and it appears to work.
    - BaseEstimator include the get/set_params methods that are required.
    - check_X_y and check_array implement checks (required by the check_estimator function)
        on the input data.
    """

    package = None
    model = None
    rng_seed: int = -1

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

    def score(self, X):
        "score a prediction in a sklearn-compatible way"
        return 0.0
