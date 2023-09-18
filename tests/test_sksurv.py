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

rsf_test = basic_test()
rsf_test.model = tosa.RSF(rng_seed=2309)
rsf_test.X, rsf_test.y = tosa.load_test_data()
rsf_test.exp_score = 0.93
rsf_test.exp_cv_mean = 0.63
rsf_test.exp_cv_std = 0.04
rsf_test.rounding = 2


def test_sksurv_data_is_loading(X=coxnet_test.X):
    assert X.shape == (198, 84)

# coxnet_test.run
@pytest.mark.parametrize("testmethod", coxnet_test.get_tests())
def test_evaluation(testmethod):
    testmethod()

# rsf_test.run
@pytest.mark.parametrize("testmethod", rsf_test.get_tests())
def test_evaluation(testmethod):
    testmethod()

