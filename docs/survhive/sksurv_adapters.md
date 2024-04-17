Module survhive.sksurv_adapters
===============================

Classes
-------

`CoxNet(rng_seed: int = None, alpha: float = None, l1_ratio: float = 0.5)`
:   Adapter for the CoxNet method from scikit-survival

    ### Ancestors (in MRO)

    * survhive.sksurv_adapters.SkSurvEstimator
    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `alpha`
    :

    `l1_ratio`
    :

    `model_`
    :

    ### Static methods

    `get_parameter_grid(max_width=None)`
    :   Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.

    ### Methods

    `predict_survival(self, X, time)`
    :

`CoxPH(rng_seed: int = None, alpha: float = 0.0, ties: str = 'efron', verbose: bool = False)`
:   Adapter for a simulated Cox Proportional Hazard (CoxPH) method from scikit-survival
    Use it only for baseline calculations, otherwise use CoxNet.

    ### Ancestors (in MRO)

    * survhive.sksurv_adapters.SkSurvEstimator
    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `alpha`
    :

    `model_`
    :

    `ties`
    :

    `verbose`
    :

    ### Static methods

    `get_parameter_grid(max_width=None)`
    :   Generate default parameter grid for optimization.
        Here max_width does nothing, it is present to keep the API uniform
        with the deep-learning-based methods.

    ### Methods

    `predict_survival(self, X, time)`
    :

`GrBoostSA(rng_seed: int = None, n_estimators: int = 100, max_depth: int = None, min_samples_split: float = 0.1, min_samples_leaf: float = 0.05, validation_fraction: float = 0.1, patience: int = 5)`
:   Adapter for the GradientBoostingSurvivalAnalysis method from scikit-survival

    ### Ancestors (in MRO)

    * survhive.sksurv_adapters.SkSurvEstimator
    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `max_depth`
    :

    `min_samples_leaf`
    :

    `min_samples_split`
    :

    `model_`
    :

    `n_estimators`
    :

    `patience`
    :

    `validation_fraction`
    :

    ### Methods

    `get_parameter_grid(self, max_width=None)`
    :   Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.

`RSF(rng_seed: int = None, n_estimators: int = 100, max_depth: int = None, min_samples_split: float = 0.1, min_samples_leaf: float = 0.05)`
:   Adapter for the RandomSurvivalForest method from scikit-survival

    ### Ancestors (in MRO)

    * survhive.sksurv_adapters.SkSurvEstimator
    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `max_depth`
    :

    `min_samples_leaf`
    :

    `min_samples_split`
    :

    `model_`
    :

    `n_estimators`
    :

    ### Methods

    `get_parameter_grid(self, max_width=None)`
    :   Generate default parameter grid for optimization
        Here max_width does nothing, it is pesent to keep the API uniform
        with the deep-learning-based methods.

`SkSurvEstimator(rng_seed: int = None)`
:   Adapter for the scikit-survival methods

    ### Ancestors (in MRO)

    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Descendants

    * survhive.sksurv_adapters.CoxNet
    * survhive.sksurv_adapters.CoxPH
    * survhive.sksurv_adapters.GrBoostSA
    * survhive.sksurv_adapters.RSF

    ### Class variables

    `model_`
    :

    `package`
    :

    `verbose`
    :

    ### Static methods

    `get_parameter_grid(max_width=None)`
    :

    ### Methods

    `predict_survival(self, X, time)`
    :