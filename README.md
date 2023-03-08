# survwrap (codename)
Convenient wrapper of various (deep) survival model

## Main objective
Create a wrapper of the most common state-of-the-art deep survival models that lets people easily test different models without having to deal with each model or library idiosyncrasies. So, no worrying about different calling conventions, data preparation (same data is good for all models), maybe also help with hyperparameter optimization.
So the plan is to create a python module as follows:
- wrap each model in a scikit-learn/scikit-survival compatible class
- uniformate the survival outcome data representation
- uniformate the survival prediction output across the models (non trivial)
- simple and well documented
- add support for hyperparameter optimization (optuna)? maybe at least a parameter grid of reasonable values for each module
- testing: test that all models run correctly on a toy dataset (continuous integration?)

## Other design choices
I would use a data representation that is enough for competing risks but no more, since few (maybe none) models can handle multi-state events.

## Paper
We also want to publich this module. In the paper we could:
- use this module on public datasets to produce a review of included models
- add simulated dataset for in depth testing about linear/non-linear, different survival models, competing risks...

## Model to be included:
- pycox models (https://github.com/havakv/pycox, last version 14/01/2022 also on pypi)
- autonlab models (https://github.com/autonlab/auton-survival, still active development, not on pypi)
- survtrace (https://github.com/RyanWangZf/SurvTRACE)?
- soden (https://github.com/jiaqima/SODEN)?
- xgboost survival
- others?

## Resources:
- implementing scikit-learn compatible models: https://scikit-learn.org/stable/developers/develop.html
- in the repo there is a ALS Benchmarks notebook that wraps some models from pycox and autonlab-survival

