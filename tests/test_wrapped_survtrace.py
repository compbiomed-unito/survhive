import pytest
import survwrap as tosa
from common import basic_test

# init test data
trace_test = basic_test()
trace_test.model = tosa.SurvTraceSingle(
    rng_seed=2310,
    num_durations=10,
    hidden_size=32,
)
trace_test.X, trace_test.y = tosa.load_test_data()
trace_test.exp_score = 0.68
trace_test.exp_cv_mean = 0.60
trace_test.exp_cv_std = 0.1
trace_test.exp_survival = [[0.60, 0.56, 0.34]]
trace_test.exp_td_harrel_score = 0.66
trace_test.exp_td_brier_score = -0.21
trace_test.exp_td_roc_auc_score = 0.79
trace_test.rounding = 2


# trace_test.run
@pytest.mark.parametrize("testmethod", trace_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
