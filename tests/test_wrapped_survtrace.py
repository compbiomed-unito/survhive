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
trace_test.exp_score = 0.67
trace_test.exp_cv_mean = 0.65
trace_test.exp_cv_std = 0.03
trace_test.exp_survival = [[0.56, 0.49, 0.28]]
trace_test.exp_td_harrel_score = 0.65
trace_test.exp_td_brier_score = 0.26
trace_test.exp_td_roc_auc_score = 0.78
trace_test.rounding = 2


# trace_test.run
@pytest.mark.parametrize("testmethod", trace_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
