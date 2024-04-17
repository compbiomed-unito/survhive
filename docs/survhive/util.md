Module survhive.util
====================

Functions
---------

    
`event_quantiles(y, quantiles=[0.25, 0.5, 0.75])`
:   get the times corresponding to the specified quantile fractions of events

    
`get_indicator(y)`
:   Get censoring indicator (bool)

    
`get_time(y)`
:   Get the time of the event

    
`load_test_data(dataset='breast_cancer')`
:   Load standard breast-cancer dataset for testing

    
`survival_crossval_splitter(X, y, n_splits=5, n_repeats=2, rng_seed=None)`
:   a RepeatedStratifiedKFold CV splitter stratified according to survival events

    
`survival_train_test_split(X, y, test_size=0.25, rng_seed=None, shuffle=True)`
:   Split survival data into train and test set using event-label stratification