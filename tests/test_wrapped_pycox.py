import pytest
import survhive
from common import basic_test

# init test data
pycox_test = basic_test()
#
pycox_test.model = survhive.DeepHitSingle(
    epochs=100,
    layer_sizes=[7, 7],
    learning_rate=0.005,
    batch_size=16,
    device="cpu",
    dropout=0.1,
    rng_seed=2308,
)
#
pycox_test.exp_score = 0.89
pycox_test.exp_cv_mean = 0.61
pycox_test.exp_cv_std = 0.01
pycox_test.exp_survival = [[0.80, 0.74, 0.65]]
pycox_test.exp_td_harrel_score = 0.89
pycox_test.exp_td_brier_score = -0.08
pycox_test.exp_td_roc_auc_score = 0.90
pycox_test.rounding = 2


# pycox_test.run
@pytest.mark.parametrize("testmethod", pycox_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
