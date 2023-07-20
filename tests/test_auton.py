import pytest
import survwrap as tosa
from common import basic_test

# init test data
dsm_test = basic_test()
dsm_test.model = tosa.DeepSurvivalMachines(
    rng_seed=2307, max_epochs=20, layer_sizes=[10, 10, 10]
)
dsm_test.X, dsm_test.y = tosa.load_test_data()
dsm_test.exp_score = 0.63
dsm_test.exp_cv_mean = 0.61
dsm_test.exp_cv_std = 0.04
dsm_test.rounding = 2


# dsm_test.run
@pytest.mark.parametrize("testmethod", dsm_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
