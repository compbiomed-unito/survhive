import pytest
import survhive
from common import basic_test

# init test data
trace_test = basic_test()
trace_test.model = survhive.SurvTraceSingle(
    rng_seed=2310,
    num_durations=10,
    # hidden_size=32,
)
trace_test.exp_score = 0.92
trace_test.exp_cv_mean = 0.61
trace_test.exp_cv_std = 0.03
trace_test.exp_survival = [[0.83, 0.75, 0.60]]
# trace_test.exp_survival = [[0.95, 0.94, 0.87]]
trace_test.exp_td_harrel_score = 0.92
trace_test.exp_td_brier_score = -0.06
trace_test.exp_td_roc_auc_score = 0.97
trace_test.rounding = 2


# trace_test.run
@pytest.mark.parametrize("testmethod", trace_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
