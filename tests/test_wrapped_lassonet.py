import pytest
import survwrap
from common import basic_test

# init test data
fastcph_test = basic_test()
fastcph_test.model = survwrap.FastCPH(
    rng_seed=2401,
    layer_sizes=[4,4,4],
    tie_approximation='efron'
    # device="cpu",
)
fastcph_test.exp_score = 0.99
fastcph_test.exp_cv_mean = 0.70
fastcph_test.exp_cv_std = 0.02
fastcph_test.exp_survival = [[1.00, 0.96, 0.60]]
# fastcph_test.exp_survival = [[0.95, 0.94, 0.87]]
fastcph_test.exp_td_harrel_score = 0.99
fastcph_test.exp_td_brier_score = -0.01
fastcph_test.exp_td_roc_auc_score = 0.99
fastcph_test.rounding = 2


# fastcph_test.run
@pytest.mark.parametrize("testmethod", fastcph_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
