# SurvHive 

SurvHive is a Python package that provides an interface to various
[survival-analysis](https://en.wikipedia.org/wiki/Survival_analysis) models,
from Cox to deep-learning-based ones, making it easier to access and compare
different methods. 

It is designed so that all method adapters are compliant
[scikit-learn](https://scikit-learn.org/stable/) estimators, allowing
interoperability with scikit-learn facilities and it offers a range of features
for model selection, parameter tuning, and performance evaluation.  

## Models provided:

From [scikit-survival](https://github.com/sebp/scikit-survival):

* CoxPH  
* CoxNet 
* Gradient Boosting Survival Analysis
* Random Survival Forest 

From [Pycox](https://github.com/havakv/pycox):

* DeepHitSingle 

From [Auton Survival](https://github.com/autonlab/auton-survival):

* Deep Survival Machines 

From [LassoNet](https://github.com/lasso-net/lassonet):

* FastCPH (extended to calculate the [Survival function](https://en.wikipedia.org/wiki/Survival_function) )

From [SurvTRACE](https://github.com/RyanWangZf/SurvTRACE):

* SurvTraceSingle 

## Metrics

SurvHive provides multiple metrics for models evaluation, such as  Harrell
C-index, Antolini score, Brier score, and AUROC, and allows for the creation of 
user-defined ones.

## Installation

See [here](INSTALL.md) for instruction.

## Getting Started

See our [guide](GETTING_STARTED.md). 

## API Documentation

See doc/ folder

## Cite

Submitted for publication

## License

This code is MIT-licensed (see included LICENSE file).
The wrapped codes are licensed according to their own terms (mostly MIT).

