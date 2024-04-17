Module survhive.optimization
============================

Functions
---------

    
`generate_topology_grid(max_width, max_layers=3)`
:   return a list of net topologies to be used in hyperparameter optimization

    
`get_grid_size(grid)`
:   calculate the number of points in a grid

    
`get_model_scores_df(search)`
:   Returns a pandas dataframe containing rank, avg_cv_score, std_cv_score,
    params for each score specified in an optimization search result.

    
`get_model_top_ranking_df(search)`
:   Returns a pandas dataframe containing the top-ranking solutions
    for each score specified in an optimization search result.
    This is a subset of what reported from the get_model_scores_df function.

    
`optimize(estimator, X, y, mode='sklearn-grid', user_grid=[], cv=None, scoring=None, tries=None, n_jobs=1, refit=True)`
:   hyperparameter optimization of estimator