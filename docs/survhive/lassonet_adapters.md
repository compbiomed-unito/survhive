Module survhive.lassonet_adapters
=================================

Classes
-------

`FastCPH(rng_seed: int = None, layer_sizes: Sequence[int] = <factory>, tie_approximation: str = 'efron', lambda_seq: Sequence[float] = <factory>, lambda_start: float = 0.001, path_multiplier: float = 1.025, backtrack: bool = False, device: str = None, verbose: int = 1, fit_lambda_: float = None)`
:   Adapter for the FastCPH method from lassonet
    
    NB: setting the parameter lambda_seq overrides the effects of
        BOTH lambda_start and path_multiplier

    ### Ancestors (in MRO)

    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `backtrack`
    :

    `device`
    :

    `fit_lambda_`
    :

    `lambda_start`
    :

    `model_`
    :

    `package`
    :

    `path_multiplier`
    :

    `rng_seed`
    :

    `tie_approximation`
    :

    `verbose`
    :

    ### Static methods

    `get_parameter_grid(max_width=None)`
    :

    ### Methods

    `predict_survival(self, X, time)`
    :