Module survhive.auton_adapters
==============================

Classes
-------

`DeepSurvivalMachines(rng_seed: int = None, n_distr: int = 2, distr_kind: str = 'Weibull', batch_size: int = 32, layer_sizes: Sequence[int] = <factory>, learning_rate: float = 0.001, validation_size: float = 0.1, max_epochs: int = 100, torch_threads: int = 0, elbo: bool = False)`
:   Adapter for the DeepSurvivalMachines method from auton-survival

    ### Ancestors (in MRO)

    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `batch_size`
    :

    `distr_kind`
    :

    `elbo`
    :

    `learning_rate`
    :

    `max_epochs`
    :

    `n_distr`
    :

    `torch_threads`
    :

    `validation_size`
    :

    ### Static methods

    `get_parameter_grid(max_width)`
    :

    ### Methods

    `fit(self, X, y)`
    :   fit an auton-survival DeepSurvivalMachines model for single events

    `predict(self, X, eval_times=None)`
    :   predict probabilites of event at given times using DeepSurvivalMachines

    `predict_survival(self, X, time)`
    :

    `set_predict_request(self: survhive.auton_adapters.DeepSurvivalMachines, *, eval_times: Union[bool, NoneType, str] = '$UNCHANGED$') -> survhive.auton_adapters.DeepSurvivalMachines`
    :   Request metadata passed to the ``predict`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``predict`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``predict``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        eval_times : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``eval_times`` parameter in ``predict``.
        
        Returns
        -------
        self : object
            The updated object.