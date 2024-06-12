import numpy
from dataclasses import dataclass
from sklearn.utils import check_X_y, check_array
from .adapter import SurvivalEstimator
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

__all__ = [
    "SkSurvEstimator",
    "CoxNet",
    "RSF",
    "CoxPH",
    "GrBoostSA",
]


class SkSurvEstimator(SurvivalEstimator):
    """
    Adapter for the scikit-survival methods
    """

    package = "scikit-survival"
    model_ = None

    # init
    verbose: bool = False

    def fit(self, X, y):
        pass

    #    def predict_survival(self, X, time):

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X)

    def predict_survival(self, X, time):
        X = check_array(X)
        return numpy.array(
            [
                numpy.interp(time, sf.x, sf.y, left=1)
                for sf in self.model_.predict_survival_function(X)
            ]
        )

    @staticmethod
    def get_parameter_grid(max_width=None):
        pass


@dataclass
class CoxNet(SkSurvEstimator):
    """
    Adapter for the CoxNet method from scikit-survival
    """

    model_ = CoxnetSurvivalAnalysis()

    # init
    alpha: float = None
    l1_ratio: float = 0.5

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_.set_params(
            alphas=None if self.alpha is None else [self.alpha],
            l1_ratio=self.l1_ratio,
            verbose=self.verbose,
            fit_baseline_model=True,
        )
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X, alpha=self.alpha)

    def predict_survival(self, X, time):
        X = check_array(X)
        return numpy.array(
            [
                numpy.interp(time, sf.x, sf.y, left=1)
                for sf in self.model_.predict_survival_function(X, alpha=self.alpha)
            ]
        )

    @staticmethod
    def get_parameter_grid(max_width=None):
        """Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.
        """

        return dict(
            alpha=[
                0.001,
                0.003,
                0.005,
                0.008,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            l1_ratio=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
        )


@dataclass
class RSF(SkSurvEstimator):
    """
    Adapter for the RandomSurvivalForest method from scikit-survival
    """

    model_ = RandomSurvivalForest()

    # init
    # l1_ratio: float = 0.5
    # fit_baseline_model: bool = False
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: float = 0.1
    min_samples_leaf: float = 0.05

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_.set_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            verbose=self.verbose,
            random_state=self.rng_seed,
        )
        self.model_ = self.model_.fit(X, y)
        return self

    def get_parameter_grid(self, max_width=None):
        """Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.
        """
        return dict(
            n_estimators=[50, 100, 200],
            max_depth=[self.max_depth],
            min_samples_split=[self.min_samples_split],
            min_samples_leaf=[self.min_samples_leaf],
        )


@dataclass
class CoxPH(SkSurvEstimator):
    """
    Adapter for a simulated Cox Proportional Hazard (CoxPH) method from scikit-survival
    Use it only for baseline calculations, otherwise use CoxNet.
    """

    model_ = CoxPHSurvivalAnalysis()

    # init
    alpha: float = 0.0
    ties: str = "efron"
    verbose: bool = False

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_.set_params(
            alpha=self.alpha,
            ties=self.ties,
            verbose=self.verbose,
        )
        self.model_ = self.model_.fit(X, y)
        return self

    def predict_survival(self, X, time):
        X = check_array(X)
        _utimes = self.model_.unique_times_
        # print ("utimes: ", _utimes.min(), _utimes.max(), _utimes[0],_utimes[-1])

        # return interpolated survival
        return numpy.array(
            [
                numpy.interp(time, _utimes, sf, left=1)
                for sf in self.model_.predict_survival_function(X, return_array=True)
            ]
        )

    @staticmethod
    def get_parameter_grid(max_width=None):
        """Generate default parameter grid for optimization.
        Here max_width does nothing, it is present to keep the API uniform
        with the deep-learning-based methods.
        """

        return dict(
            alpha=[
                0.001,
                0.003,
                0.005,
                0.008,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
            ],
        )


@dataclass
class GrBoostSA(SkSurvEstimator):
    """
    Adapter for the GradientBoostingSurvivalAnalysis method from scikit-survival
    """

    model_ = GradientBoostingSurvivalAnalysis()

    # init
    # l1_ratio: float = 0.5
    # fit_baseline_model: bool = False
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: float = 0.1
    min_samples_leaf: float = 0.05
    validation_fraction: float = 0.1
    patience: int = 5

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_.set_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.patience,
            verbose=self.verbose,
            random_state=self.rng_seed,
        )
        self.model_ = self.model_.fit(X, y)
        return self

    def get_parameter_grid(self, max_width=None):
        """Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.
        """
        return dict(
            n_estimators=[
                50,
                100,
                200,
            ],
            max_depth=[self.max_depth],
            min_samples_split=[self.min_samples_split / _ for _ in [1, 2]],
            min_samples_leaf=[self.min_samples_leaf / _ for _ in [1, 2, 5]],
            patience=[None, self.patience],
        )
