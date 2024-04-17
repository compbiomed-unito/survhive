Module survhive.adapter
=======================

Classes
-------

`SurvivalEstimator(rng_seed: int = None)`
:   This is a minimal (empty) estimator that passes the sk-learn check_estimator tests.
    
    Dataclasses can be useful to avoid long init functions and it appears to work.
    - BaseEstimator include the get/set_params methods that are required.
    - check_X_y and check_array implement checks (required by check_estimator function)
        on the input data.

    ### Ancestors (in MRO)

    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Descendants

    * survhive.auton_adapters.DeepSurvivalMachines
    * survhive.lassonet_adapters.FastCPH
    * survhive.pycox_adapters.DeepHitSingle
    * survhive.sksurv_adapters.SkSurvEstimator
    * survhive.survtrace_adapters.SurvTraceSingle

    ### Class variables

    `model`
    :

    `package`
    :

    `rng_seed`
    :

    ### Methods

    `fit(self, X, y)`
    :   fit the model

    `predict(self, X)`
    :   do a prediction using a fit model

    `predict_survival(self, X, times)`
    :

    `score(self, X, y)`
    :   return the Harrell's c-index as a sklearn score