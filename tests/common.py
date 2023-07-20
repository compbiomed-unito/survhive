import pytest
import survwrap as tosa
import numpy as np
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, field
from sklearn.utils.estimator_checks import check_estimator

test_X, test_y = tosa.load_test_data()


class basic_test:
    """Basic test for model:
    Given a parameterized model, its test data, and the expected results
    1. fits the modelaccording to the hyperparameter specs;
    2. calculates the score (Harrel C-index) of the fit and compares to the expected values;
    3. performs a 3-fold cross validation of the fit model;
    4. calculates mean and std for the CV scores and compares them to their expected values.
    """

    model: field(default_factory=tosa.SurvivalEstimator)
    X: field(default_factory=test_X)
    y: field(default_factory=test_y)
    exp_score: float
    exp_cv_mean: float
    exp_cv_std: float
    rounding: int

    def get_tests(self):
        return [
            getattr(self, testname)
            for testname in dir(self)
            if testname.startswith("test_")
        ]

    def test_sklearn_compliance(self):
        "Check adapter for scikit-learn compliance"
        try:
            check_estimator(self.model)
        except Exception as e:
            print(e.__class__.__name__)
            print(e)

    def test_fit_score(self):
        "assert on prediction score"
        fit_score = self.model.fit(self.X, self.y).score(self.X, self.y)
        assert fit_score.round(self.rounding) == self.exp_score

    def test_cv(self):
        "assert on 3-fold average CV score and its std. deviation"
        cv_score = cross_val_score(self.model, self.X, self.y, cv=3)
        cv_avg_score = np.array([cv_score.mean(), cv_score.std()])
        np.testing.assert_array_almost_equal(
            cv_avg_score, [self.exp_cv_mean, self.exp_cv_std], decimal=self.rounding
        )


# # init test data example
# dsm_test = basic_test()
# dsm_test.model = tosa.DeepSurvivalMachines(
#     rng_seed=2307, max_epochs=20, layer_sizes=[10, 10, 10]
# )
# dsm_test.X = test_X
# dsm_test.y = test_y
# dsm_test.exp_score = 0.63
# dsm_test.exp_cv_mean = 0.61
# dsm_test.exp_cv_std = 0.04
# dsm_test.rounding = 2
#
# # dsm_test.run
# @pytest.mark.parametrize("testmethod", dsm_test.get_tests())
# def test_evaluation(testmethod):
#     testmethod()
