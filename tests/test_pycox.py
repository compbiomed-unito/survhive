import pytest
import survwrap as tosa
from common import basic_test

# init test data
pycox_test = basic_test()
pycox_test.X, pycox_test.y = tosa.load_test_data()
#
pycox_test.model = tosa.DeepHitSingle(
    epochs=100,
    #      layer_sizes=[10, 10],
    learning_rate=0.05,
    batch_size=len(pycox_test.y) // 2,
    device="cpu",
    rng_seed=2307,
)
#
pycox_test.exp_score = 0.96
pycox_test.exp_cv_mean = 0.56
pycox_test.exp_cv_std = 0.09
pycox_test.rounding = 2


# pycox_test.run
@pytest.mark.parametrize("testmethod", pycox_test.get_tests())
def test_evaluation(testmethod):
    testmethod()
