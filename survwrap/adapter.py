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

    param1: int = (1,)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._validate_data(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return np.full(shape=X.shape[0], fill_value=self.param1)


@dataclass
class CoxNet(SurvivalEstimator):
    """
    Adapter for the CoxNet method from scikit-survival
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    package = "scikit-survival"
    model_ = CoxnetSurvivalAnalysis()

    # init
    l1_ratio: float = 0.5
    verbose: bool = False
    fit_baseline_model: bool = False
    score: float = 0.0

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.model_.set_params(
            l1_ratio=self.l1_ratio,
            verbose=self.verbose,
            fit_baseline_model=self.fit_baseline_model
            )
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X)

    def score(self, X, y):
        X = check_array(X)
        return self.model_.score(X, y)



