Module survhive.pycox_adapters
==============================

Classes
-------

`DeepHitSingle(rng_seed: int = None, num_durations: int = 10, layer_sizes: Sequence[int] = <factory>, epochs: int = 100, batch_size: int = 64, validation_size: float = 0.1, learning_rate: float = 0.001, dropout: float = 0.2, device: str = 'cpu', verbose: bool = False)`
:   Adapter for the DeepHitSingle method from pycox

    ### Ancestors (in MRO)

    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `batch_size`
    :

    `device`
    :

    `dropout`
    :

    `epochs`
    :

    `learning_rate`
    :

    `model_`
    :

    `num_durations`
    :

    `package`
    :

    `validation_size`
    :

    `verbose`
    :

    ### Static methods

    `get_parameter_grid(max_width)`
    :

    ### Methods

    `fit(self, X, y)`
    :   fit a Pycox DeepHit model for single events

    `predict_survival(self, X, time)`
    :