import pytest
import survwrap as tosa
from common import basic_test

# init test data
coxnet_test = basic_test()
coxnet_test.model = tosa.CoxNet(rng_seed=2307)
coxnet_test.X, coxnet_test.y = tosa.load_test_data()
coxnet_test.exp_score = 0.95
coxnet_test.exp_cv_mean = 0.63
coxnet_test.exp_cv_std = 0.04
coxnet_test.rounding = 2


# coxnet_test.run
@pytest.mark.parametrize("testmethod", coxnet_test.get_tests())
def test_evaluation(testmethod):
    testmethod()


def test_sksurv_data_is_loading(X=coxnet_test.X):
    assert X.shape == (198, 84)
