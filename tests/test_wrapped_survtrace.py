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
trace_test.exp_score = 0.92
trace_test.exp_cv_mean = 0.65
trace_test.exp_cv_std = 0.01
trace_test.rounding = 2


# trace_test.run
@pytest.mark.parametrize("testmethod", trace_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
