from __future__ import annotations

import numpy
import torch
from dataclasses import dataclass, field
from collections.abc import Sequence  # abc: Abstract Base Class
from sklearn.utils import check_X_y, check_array
from sksurv.metrics import concordance_index_censored

# from auton_survival.models.dsm import DeepSurvivalMachines
from .adapter import SurvivalEstimator
from .util import get_time, get_indicator
from .optimization import generate_topology_grid
import auton_survival

# __all__ = ['DeepSurvivalMachines']


@dataclass
class DeepSurvivalMachines(SurvivalEstimator):
    "Adapter for the DeepSurvivalMachines method from auton-survival"
    n_distr: int = 2
    distr_kind: str = "Weibull"
    batch_size: int = 32
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    learning_rate: float = 0.001
    validation_size: float = 0.1
    max_epochs: int = 100
    elbo: bool = False  # what is this?

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def fit(self, X, y):
        "fit an auton-survival DeepSurvivalMachines model for single events"

        # inits and checks
        self._seed_rngs()
        X, y = check_X_y(X, y)
        # calculate median time-of-event for the training set.
        # Used in default prediction
        # self.median_time_ = numpy.median(X)
        self.median_time_ = numpy.median(get_time(y))

        # fit
        self.model_ = auton_survival.DeepSurvivalMachines(
            k=self.n_distr,
            distribution=self.distr_kind,
            layers=self.layer_sizes,
        ).fit(
            X,
            get_time(y),
            get_indicator(y),
            learning_rate=self.learning_rate,
            vsize=self.validation_size,
            batch_size=self.batch_size,
            iters=self.max_epochs,  # this should be the maximum number of epochs
            elbo=self.elbo,
        )
        return self

    def predict(self, X, eval_times=None):
        """predict probabilites of event at given times using DeepSurvivalMachines"""
        X = check_array(X)
        # set default time of prediction at training median
        if eval_times is None:
            self.single_event = True
        eval_times = [self.median_time_] if self.single_event else eval_times

        _preds = numpy.swapaxes(
            [self.model_.predict_risk(X, t)[:, 0] for t in eval_times], 0, 1
        )
        return _preds.flatten() if self.single_event else _preds

    def harrell_score(self, y_true, y_pred, *args, **kwargs):
        "return Harrell's C-index for a prediction"

        return concordance_index_censored(
            # event_indicator=y_true[y_true.dtype.names[0]],
            event_indicator=get_indicator(y_true),
            event_time=get_time(y_true),
            estimate=y_pred,
            *args,
            **kwargs,
        )

    def score(self, X, y):
        "return the Harrell's c-index as a sklearn score"
        X, y = check_X_y(X, y)
        return self.harrell_score(y, self.predict(X))[0]

    @staticmethod
    def get_parameter_grid(max_width):
        return dict(
            n_distr=[1, 2, 3],
            distr_kind=["Weibull"],
            batch_size=[16, 32],
            layer_sizes=generate_topology_grid(max_width),
            learning_rate=[0.005, 0.001],
            validation_size=[0.1],
            max_epochs=[100],
            elbo=[False],
        )
