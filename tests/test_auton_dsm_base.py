import survwrap as tosa
import numpy as np
from sklearn.model_selection import cross_val_score

test_X, test_y = tosa.load_test_data()

def test_wrapped_DSM(X=test_X, y=test_y):
    "fit a DeepSurvivalMachines model (from auton-survival)"
    model = tosa.DeepSurvivalMachines(rng_seed=2307, max_epochs=20, layer_sizes=[10,10,10])
    model.fit(X, y)
    pred = model.predict(X)
    fit_score = model.score(X, y)

    # assert on simple score
    assert fit_score.round(3) == 0.626

    # assert on 3-fold cross-validation score
    cv_score = cross_val_score(model, X, y, cv=3)
    # testing assertion: arrays uguali alla 3° decimale
    np.testing.assert_array_almost_equal(cv_score, [0.649, 0.615, 0.564], decimal=3)
    # testing on average CV score ( mean, std ), two decimals
    cv_avg_score = np.array([cv_score.mean(), cv_score.std()]) 
    np.testing.assert_array_almost_equal(cv_avg_score, 
            [ 0.61, 0.04 ], 
            decimal=2)




