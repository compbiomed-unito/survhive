import pytest
import survwrap
from common import basic_test

# init test data
dsm_test = basic_test()
dsm_test.model = survwrap.DeepSurvivalMachines(
    rng_seed=2307,
    torch_threads=2,
    batch_size=10,
    max_epochs=20,
    layer_sizes=[10, 10, 10],
)
dsm_test.exp_score = 0.91
dsm_test.exp_cv_mean = 0.64
dsm_test.exp_cv_std = 0.10
dsm_test.exp_survival = [[0.91, 0.87, 0.79]]
dsm_test.exp_td_harrel_score = 0.91
dsm_test.exp_td_brier_score = -0.08
dsm_test.exp_td_roc_auc_score = 0.94
dsm_test.rounding = 2


# dsm_test.run
@pytest.mark.parametrize("testmethod", dsm_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
