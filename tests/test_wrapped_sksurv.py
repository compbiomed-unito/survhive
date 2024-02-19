import pytest
import survhive
from common import basic_test

# init test data
coxnet_test = basic_test()
coxnet_test.model = survhive.CoxNet(
    rng_seed=2401,
    alpha=0.015,
)
coxnet_test.exp_score = 0.92
coxnet_test.exp_cv_mean = 0.64
coxnet_test.exp_cv_std = 0.02
coxnet_test.exp_survival = [[0.89, 0.71, 0.52]]
coxnet_test.exp_td_harrel_score = 0.92
coxnet_test.exp_td_brier_score = -0.04
coxnet_test.exp_td_roc_auc_score = 0.96
coxnet_test.rounding = 2

rsf_test = basic_test()
rsf_test.model = survhive.RSF(rng_seed=2309)
rsf_test.exp_score = 0.98
rsf_test.exp_cv_mean = 0.63
rsf_test.exp_cv_std = 0.01
rsf_test.exp_survival = [[0.9, 0.79, 0.67]]
rsf_test.exp_td_harrel_score = 0.98
rsf_test.exp_td_brier_score = -0.07
rsf_test.exp_td_roc_auc_score = 0.99
rsf_test.rounding = 2

coxph_test = basic_test()
coxph_test.model = survhive.CoxPH(rng_seed=2311, alpha=0.1)
# coxph_test.X, coxph_test.y = survhive.load_test_data()
coxph_test.X = coxph_test.X
coxph_test.exp_score = 0.94
coxph_test.exp_cv_mean = 0.63
coxph_test.exp_cv_std = 0.04
coxph_test.exp_survival = [[0.98, 0.82, 0.49]]
coxph_test.exp_td_harrel_score = 0.94
coxph_test.exp_td_brier_score = -0.02
coxph_test.exp_td_roc_auc_score = 0.98
coxph_test.rounding = 2


def test_sksurv_data_is_loading(X=coxnet_test.X):
    assert X.shape == (198, 84)


# coxnet_test.run
@pytest.mark.parametrize("testmethod", coxnet_test.get_tests())
def test_evaluation_coxnet(testmethod):
    testmethod()


# rsf_test.run
@pytest.mark.parametrize("testmethod", rsf_test.get_tests())
def test_evaluation_rsf(testmethod):
    testmethod()


# coxph_test.run
@pytest.mark.parametrize("testmethod", coxph_test.get_tests())
def test_evaluation_coxph(testmethod):
    testmethod()
