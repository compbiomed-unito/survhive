from __future__ import annotations

import numpy
import torch
from dataclasses import dataclass, field
from collections.abc import Sequence  # abc: Abstract Base Class
from sklearn.utils import check_X_y, check_array

from .adapter import SurvivalEstimator
from .util import get_time, get_indicator
from .optimization import generate_topology_grid
import auton_survival

__all__ = [
    "DeepSurvivalMachines",
]


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
    torch_threads: int = 0
    elbo: bool = False  # what is this?

    def _limit_torch_threads(self):
        "limit the maximum number of Torch CPU threads"
        if self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed and self.rng_seed > 0:
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
        self._limit_torch_threads()

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

    def predict_survival(self, X, time):
        X = check_array(X)
        try:
            n_times = len(time)
            # for some reason only a list is treated properly as multiple times,
            # an array yield a prediction of shape (len(X), 1)
            if not isinstance(time, list):
                time = time.tolist()  # for some
        except TypeError:
            n_times = 0
        r = self.model_.predict_survival(X, time)
        if n_times == 0:
            assert r.shape == (len(X), 1)
            r = r[:, 0]
        return r

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

    @staticmethod
    def get_parameter_grid(max_width: int):
        """set-up a default grid for parameter optimization.

        Arguments:
            max_width: the dimension of a neural-network layer

        Returns:
            a dictionary of parameters compatible with sklearn search methods
        """
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
