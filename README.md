# survwrap
Convenient wrapper of various (deep) survival model

Main objective
Create a wrapper of the most common state-of-the-art deep survival models that lets people easily test different models without having to deal with each model or library idiosyncrasies. So, no worrying about different calling conventions, data preparation (same data is good for all models), maybe also help with hyperparameter optimization.
So the plan is:
- wrap each model in a scikit-learn/scikit-survival compatible class
- uniformate the survival outcome data representation
- 
- add support for hyperparameter optimization (optuna)? maybe at least a parameter grid of reasonable values for each module

For the paper:
- use this module on public datasets to produce a review of included models
- add simulated dataset for in depth testing about linear/non-linear, different survival models, competing risks...

Model to be included:
- pycox models
- autonlab models
- survtrace?
- others?

I would use a data representation that is enough for competing risks but no more, since few (maybe none) models can handle multi-state events.
