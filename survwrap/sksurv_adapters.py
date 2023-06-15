from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array
from .adapter import SurvivalEstimator


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

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_.set_params(
            l1_ratio=self.l1_ratio,
            verbose=self.verbose,
            fit_baseline_model=self.fit_baseline_model,
        )
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X)

    def score(self, X, y):
        X, y = check_X_y(X, y)
        return self.model_.score(X, y)
