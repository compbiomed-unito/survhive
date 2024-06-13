
## Getting started with SurvHive

This guide is intended as a simple and basic introductory overview to
SurvHive usage. It is not comprehensive of all the SurvHive
functionalities but focusses on how to make use of the most important
SurvHive features.

For detailed reference documentation of the functions and classes
contained in the package, see the GitHub page:
<https://github.com/compbiomed-unito/survhive>.

## Introduction

Here, we present SurvHive, a wrapper library for Survival Analysis,
designed to wrap up several of the best methods for survival data.
Survival Analysis is a robust statistical method used to examine and
predict the time until the occurrence of an event of interest, such as
patient death, equipment failure or customer churn.

The package provides access to eight different models coming from
different packages, namely:

-   CoxPH, CoxNet, Gradient Boosting Survival Analysis (GRBoostSA) and
    Random Survival Forest (RSF) from the [Scikit-Survival][@polsterl2020scikit] package;

-   DeepHitSingle from the [PyCox][@geck2012pycox] package;

-   Deep Survival Machines from the [Auton Survival][@nagpal2022auton]
    package;

-   FastCPH from the [LassoNet][@lemhadri2021lassonet] package;

-   SurvTraceSingle from the [SurvTRACE][@wang2022survtrace] package.

## Installing Jupyter Notebook and SurvHive

When evaluating your data, you have to provide a correct .csv/.xlsx
file. It is necessary to create the correct environment in order to
process a .csv/.xlsx file or a DataFrame preprocessed. SurvHive can be
used in different ways:

-   Anaconda installed and Jupyter Notebook.

-   Jupyter Notebook without Anaconda

-   Using the library in a script.

However, in the following we assume, for sake of simplicity, to use
SurvHive on a notebook. At this link can be founded all the information
for the installation of Jupyter: <https://jupyter.org/install>.

After this step, we can start with the library. In order to install
SurvHive, first of all you need to have a Conda environment or a
virtualenv activate. Thus, from a notebook cell, we can use the
following command:

``` {.python linenos="" frame="lines" fontsize="\\small"}
!pip install survhive
```

If you prefer to install the library from a terminal, you need only to remove
the exclamation mark \"!\". The [installation guide for Conda](INSTALL.md#installing-with-conda) 
is provided too.  

<!-- <https://github.com/compbiomed-unito/survhive/blob/main/INSTALL.md> -->

## Import Dataset and Prepare Splits

Let's start with your .csv file, that must have two columns,
'event','event_time'. The first column indicates if the event is true
(value 1) or censored (valued 0). The latter ('event_time'), contains
the time of the events, in the same temporal unit.

You can upload your dataset as .csv file into a DataFrame and by
importing the SurvHive library:

``` {.python linenos="" frame="lines" fontsize="\\small"}
import pandas as pd
import survhive as sv

dataset = pd.read_csv('example_dataset.csv', index_col='id')
```

After this step we can get the feature matrix and the label vector. Just
run the following command from [Scikit-Survival][@polsterl2020scikit]:

``` {.python linenos="" frame="lines" fontsize="\\small"}
from sksurv.datasets import get_x_y
X, y = get_x_y(dataset, 
               attr_labels=['event','event_time'], 
               pos_label=True)
```

In the attr_labels in the code above, insert the two columns that
correspond to the label of the event and the event time.

In order to create the training and test split, we can use a simple
command in SurvHive:

``` {.python linenos="" frame="lines" fontsize="\\small"}
X_train, X_test, y_train, y_test = sv.survival_train_test_split(X, 
                                                                y, 
                                                                rng_seed=42)
```

Now, remember that you can impute and scale (if it is necessary) your data,
this can be useful because some models from different libraries don't accept
NaN data. For example, it can be done with two other functions from
[Scikit-Learn][@skl] :

``` {.python linenos="" frame="lines" fontsize="\\small"}
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer().fit(X_train)
[X_train, X_test] = [imputer.transform(_) for _ in [X_train, X_test]]

scaler = StandardScaler().fit(X_train)
[X_train, X_test] = [scaler.transform(_) for _ in [X_train, X_test]]
```

Now, we are ready for the models.

## Models from different Libraries

We can call different model in the following way. In order to test the
different performances we can also initialize some dictionaries and a
class useful later:

``` {.python linenos="" frame="lines" fontsize="\\small"}
vanilla_mods= {}
antolini_vanilla_test = {}

def model_class(_model):
    return type(_model).__name__

vanilla_mods['CoxNet'] = sv.CoxNet(rng_seed=seed)
vanilla_mods['GrBoostSA'] = sv.GrBoostSA(rng_seed=seed)
vanilla_mods['SurvTraceSingle'] = sv.SurvTraceSingle(rng_seed=seed)
```

Now, the training and test are simple:

``` {.python linenos="" frame="lines" fontsize="\\small"}
for _ in vanilla_mods.keys():
    vanilla_mods[_].fit(X_train, y_train)
    antolini_vanilla_test[_] = vanilla_mods[_].score(X_test, y_test)
```

At the end, you will have a dictionary **antolini_vanilla_test** with
all your results.

## Parameter Optimization

We can perform a parameter optimization in order to find the best model.
We can start with a random search on the model parameter space. Each
optimization step does a two repeats of 5-fold cross validations. The
score is the average on the internal cross validated tests.

``` {.python linenos="" frame="lines" fontsize="\\small"}
from IPython.display import display

for _model in vanilla_mods.keys():
    print("Optimizing", _model)
    # perform some points of random search on default search grid
    opt_m, opt_p, opt_m_s = sv.optimize(vanilla_mods[_model],
                                        X_train, y_train, 
                                        mode='sklearn-random', tries=10, 
                                        n_jobs=4)
    print("Top-ranking model for", _model)
    # N.B. mean_test_score is the mean of the cross-validation
    # runs on the *training* data
    display(sv.get_model_top_ranking_df(opt_m_s))
    
    print("Top-ten models for", _model)
    display(sv.get_model_scores_df(opt_m_s)[:10])
    antolini_best_test[_model]=_opt_model.score(X_test,y_test)
```

As a result, it is possible to extract the best models and performances
and create a DataFrame to save them:

``` {.python linenos="" frame="lines" fontsize="\\small"}
optimized_df = pandas.DataFrame(data=antolini_best_test.values(),
                                index=antolini_best_test.keys(),
                                columns=['Somewhat optimized'])
```

## Other Metrics

The '.score' method evaluate the Antolini Concordance Index. We can
evaluate other scores and with this command you can see all the scorer
available:

``` {.python linenos="" frame="lines" fontsize="\\small"}
sv.get_scorer_names()
```

Until now, you can use one of the following from the SurvHive library:

-   'c-index-antolini',

-   'c-index-quartiles',',

-   'c-index-deciles',

-   'roc-auc-quartiles',

-   'roc-auc-deciles',

-   'neg-brier-quartiles',

-   'neg-brier-deciles'

``` {.python linenos="" frame="lines" fontsize="\\small"}
brier = sv.get_scorer('neg-brier-quartiles')

for _ in vanilla_mods.keys():
    vanilla_mods[_].fit(X_train, y_train)
    antolini_vanilla_test[_] = brier(vanilla_mods[_], X_test, y_test)
```

Of course, there are always available the other metrics from
[scikit-learn][@skl], there is a method in SurvHive that can adapt every
metric. For example for the Matthews Correlation Coefficient
``` {.python linenos="" frame="lines" fontsize="\\small"}
from sklearn.metrics import matthews_corrcoef

mcc = sv.make_survival_scorer(matthews_corrcoef)
```

## Final remark

For further details on SurvHive's functionalities, users are encouraged
to consult the reference documentation available on the SurvHive GitHub
page <https://github.com/compbiomed-unito/survhive> and the .html files
in the **docs/** folder.

We hope that you will find Survhive a useful tool.


[@polsterl2020scikit]: https://github.com/sebp/scikit-survival
[@geck2012pycox]: https://github.com/havakv/pycox
[@nagpal2022auton]: https://github.com/autonlab/auton-survival
[@lemhadri2021lassonet]: https://github.com/lasso-net/lassonet
[@wang2022survtrace]: https://github.com/RyanWangZf/SurvTRACE
[@skl]: https://scikit-learn.org/stable/

