Module survhive.survtrace_adapters
==================================

Classes
-------

`SurvTraceSingle(rng_seed: int = None, num_durations: int = 5, horizons: Sequence[float] = <factory>, hidden_factor: int = 4, intermediate_size: int = 64, num_hidden_layers: int = 3, num_attention_heads: int = 2, validation_size: float = 0.1, hidden_dropout: float = 0.0, attention_dropout: float = 0.1, patience: int = 5, batch_size: int = 64, epochs: int = 100, device: str = None)`
:   Adapter for the SurvTraceSingle method from SurvTRACE

    ### Ancestors (in MRO)

    * survhive.adapter.SurvivalEstimator
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Class variables

    `attention_dropout`
    :

    `batch_size`
    :

    `device`
    :

    `epochs`
    :

    `hidden_dropout`
    :

    `hidden_factor`
    :

    `intermediate_size`
    :

    `model_`
    :

    `num_attention_heads`
    :

    `num_durations`
    :

    `num_hidden_layers`
    :

    `package`
    :

    `patience`
    :

    `rng_seed`
    :

    `validation_size`
    :

    ### Static methods

    `get_parameter_grid(max_width=None)`
    :

    ### Methods

    `predict_survival(self, X, time)`
    :

    `set_predict_request(self: survhive.survtrace_adapters.SurvTraceSingle, *, eval_times: Union[bool, NoneType, str] = '$UNCHANGED$') -> survhive.survtrace_adapters.SurvTraceSingle`
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