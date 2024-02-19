#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy


# ## Mini example for scikit-survival estimators


def load_toy_data():
    "Load standard dataset for testing"
    import sksurv.datasets
    from sklearn.preprocessing import OneHotEncoder

    X, y = sksurv.datasets.load_breast_cancer()
    X = numpy.concatenate(
        [
            X.select_dtypes("float"),
            OneHotEncoder(sparse=False).fit_transform(X.select_dtypes("category"))
            # OneHotEncoder(sparse_output=False).fit_transform(X.select_dtypes('category'))
        ],
        axis=1,
    )
    return X, y


# In[3]:


X, y = load_toy_data()
X.shape


# Fit penalized Cox model (from scikit-survival)

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

model = CoxnetSurvivalAnalysis()
model.fit(X, y)
pred = model.predict(X)

# Standard functions from scikit-learn can be used with scikit-survival models

from sklearn.model_selection import cross_val_score

cross_val_score(CoxnetSurvivalAnalysis(), X, y)


# ## Scikit-learn compatibility
# Scikit-learn has a checker for estimators to see if they conform to their specification.

# In[8]:


from sklearn.utils.estimator_checks import check_estimator


# Scikit-survival models do not necessarily pass ;-)

# In[9]:


try:
    check_estimator(CoxnetSurvivalAnalysis())
except Exception as e:
    print(e.__class__.__name__)
    print(e)


# This is an example of a minimal (empty) estimator that passes the tests. Dataclasses can be useful to avoid long __init__ functions and it appears to work. BaseEstimator include the get/set_params methods that are required. check_X_y and check_array implement checks (required by the check_estimator function) on the input data.

# In[10]:


from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array


@dataclass
class TestEstimator(BaseEstimator):
    param1: int = (1,)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._validate_data(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return numpy.full(shape=X.shape[0], fill_value=self.param1)


check_estimator(TestEstimator(param1=33))


# ## Outcome format

# Esempio formato scikit-survival

# In[11]:


y[:10]


# In[12]:


def get_indicator(y):
    return y[y.dtype.names[0]]


def get_time(y):
    return y[y.dtype.names[1]]


# In[13]:


get_indicator(y[:10])


# ## Wrapper for the DeepHit method from the pycox module

# Molto preliminare, sto recuperando e adattando il codice dal notebook ALS

# In[14]:


import pycox


# In[26]:


from dataclasses import dataclass, field
from sklearn.base import BaseEstimator
from collections.abc import Sequence


@dataclass
class DeepHitPycox(BaseEstimator):
    num_durations: int = 10
    # qui mettiamo i parametri per la forma della rete, cercherei di fare qualcosa che rispetti il paper originale
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    epochs: int = 10  # maybe implement also early stopping
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = "cpu"

    def fit(self, X, y):
        from pycox.models import DeepHitSingle
        import torchtuples as tt

        # X, y = check_X_y(X, y)
        optimizer = tt.optim.AdamWR(
            lr=self.learning_rate,
            # decoupled_weight_decay=,
            # cycle_eta_multiplier=,
        )
        self.labtrans_ = DeepHitSingle.label_transform(self.num_durations)
        y_discrete = self.labtrans_.fit_transform(get_time(y), get_indicator(y))
        net = tt.practical.MLPVanilla(
            in_features=X.shape[1],
            out_features=self.labtrans_.out_features,
            num_nodes=self.layer_sizes,
            # batch_norm, dropout,
            # **self.model_params['indepnet'], **self.model_params['net']
        )
        self.model_ = DeepHitSingle(
            net,
            optimizer,
            device=self.device,
        )

        self.median_time_ = numpy.median(get_time(y))
        self.model_.fit(
            X.astype("float32"),
            y_discrete,
            num_workers=0 if True else n_jobs,
            verbose=False,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        return self

    def predict(self, X, eval_times=None):
        if eval_times is None:
            eval_times = [self.median_time_]
        preds = 1 - self.model_.predict_surv(X.astype("float32"))
        # print('predict', eval_times.shape, self.labtrans_.cuts.shape, preds.shape)
        return numpy.array(
            [
                numpy.interp(eval_times, self.labtrans_.cuts, p, left=0, right=1)
                for p in preds
            ]
        )


preds = DeepHitPycox(epochs=3).fit(X, y).predict(X)
preds


# In[ ]:


# In[23]:


def concordance_index_score(y_true, y_pred, *args, **kwargs):
    print(y_true)
    from sksurv.metrics import concordance_index_censored

    return concordance_index_censored(
        event_indicator=get_indicator(y_true),
        event_time=get_time(y_true),
        estimate=y_pred,
        *args,
        **kwargs,
    )


# In[24]:


cross_val_score(DeepHitPycox(), X, y, scoring=concordance_index_score)


# # Pasticci

# In[16]:


from sklearn.tree import DecisionTreeClassifier


# In[17]:


cross_val_score(DecisionTreeClassifier(), X, get_indicator(y))


# In[18]:


class DeepSurvivalMachines:
    def fit(self, X, y):
        import sys

        sys.path.append("./auton-survival")
        from auton_survival.models.dsm import DeepSurvivalMachines

        try:
            self.model_ = DeepSurvivalMachines(**self.model_params["mod"]).fit(
                X, times, events, **self.model_params["fit"]
            )
        except RuntimeError as e:
            raise FailedModel(f"{self.short_name} model fit failed: {e}")
        return self

    def predict(self, X, eval_times):
        """predict probabilites of event up to given times for each event"""
        # global dbg
        # dbg = self.model_, X, eval_times, numpy.swapaxes([self.model_.predict_risk(X, t)[:, 0] for t in eval_times], 0, 1)
        return numpy.swapaxes(
            [self.model_.predict_risk(X, t)[:, 0] for t in eval_times], 0, 1
        )
        # return numpy.nan_to_num(numpy.swapaxes([self.model_.predict_risk(X, t)[:, 0] for t in eval_times], 0, 1), nan=0.5, posinf=1, neginf=0)


DeepSurvivalMachines().fit(X, y).predict(X)


# In[ ]:


cross_val_score(CoxnetSurvivalAnalysis(), X, y, scoring=concordance_index_score)


# In[ ]:


import survhive


# In[ ]:
