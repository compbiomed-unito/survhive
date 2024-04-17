Module survhive.datasets
========================
Some frequently used benchmark datasets for survival analysis

Functions
---------

    
`get_data(set_name)`
:   Load one of the available benchmark datasets as a dataset object

    
`list_available_datasets()`
:   list the available benchmark datasets

Classes
-------

`dataset(name: str, dataframe: Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7f77adda5f10>,default_factory=<class 'pandas.core.frame.DataFrame'>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),_field_type=None))`
:   a dataset container

    ### Methods

    `discard_zero_times(self)`
    :   In-place (side effects!) discard data points with zero times.
        Returns the new shape of the dataset.

    `get_X_y(self)`
    :   return dataset as two numpy ndarrays

    `index_zero_times(self)`
    :   Return pandas indexes of event with a zero time.
        Usually these data points should be removed.
        Removal can be performed *inplace* using the discard_zero_times method.