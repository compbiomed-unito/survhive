from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.utils import check_X_y, check_array

import pycox.models as Pycox
import numpy
import torch
import torchtuples as tt

from .adapter import SurvivalEstimator
from .util import (
    get_time,
    get_indicator,
    survival_train_test_split,
)
from .optimization import generate_topology_grid

__all__ = [
    "DeepHitSingle",
]


@dataclass
class DeepHitSingle(SurvivalEstimator):
    """
    Adapter for the DeepHitSingle method from pycox
    """

    package = "pycox"
    model_ = SurvivalEstimator()

    # init
    num_durations: int = 10
    # qui mettiamo i parametri per la forma della rete,
    # cercherei di fare qualcosa che rispetti il paper originale
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    epochs: int = 100
    batch_size: int = 64
    validation_size: float = 0.1
    learning_rate: float = 0.001
    dropout: float = 0.2
    device: str = "cpu"
    verbose: bool = False

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed and self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def fit(self, X, y):
        "fit a Pycox DeepHit model for single events"

        # from pycox.models import DeepHitSingle

        self._seed_rngs()
        X, y = check_X_y(X, y)
        # generate internal validation set for early-stopping
        X_train, X_val, y_train, y_val = survival_train_test_split(
            X, y, test_size=self.validation_size, rng_seed=self.rng_seed
        )
        optimizer = tt.optim.AdamWR(
            lr=self.learning_rate,
            # decoupled_weight_decay=,
            # cycle_eta_multiplier=,
        )
        # discretize on train and transform both train and val
        self.labtrans_ = Pycox.DeepHitSingle.label_transform(self.num_durations)
        y_train_discrete = self.labtrans_.fit_transform(
            get_time(y_train), get_indicator(y_train)
        )
        y_val_discrete = self.labtrans_.transform(get_time(y_val), get_indicator(y_val))

        net = tt.practical.MLPVanilla(
            in_features=X.shape[1],
            out_features=self.labtrans_.out_features,
            num_nodes=self.layer_sizes,
            batch_norm=True,
            dropout=self.dropout,
            # **self.model_params['indepnet'], **self.model_params['net']
        )
        self.model_ = Pycox.DeepHitSingle(
            net,
            optimizer,
            device=self.device,
        )

        self.median_time_ = numpy.median(get_time(y))
        # setup callback for early stopping
        _callbacks = [tt.callbacks.EarlyStopping()]
        # BIG FAT WARNING: fit returns a TrainingLogger, not a fitted model.
        # there are side effects on model_ itself
        self.training_log_ = self.model_.fit(
            X_train.astype("float32"),
            y_train_discrete,
            # num_workers=0 if True else n_jobs,
            verbose=self.verbose,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=_callbacks,
            val_data=(X_val.astype("float32"), y_val_discrete),
        )
        return self

    def _interpolate_prediction(self, method_name, X, time, left, right):
        X = check_array(X).astype("float32")
        try:
            n_times = len(time)
        except TypeError:
            n_times = 0
        pred = getattr(self.model_, method_name)(X)
        r = numpy.array(
            [
                numpy.interp(time, self.labtrans_.cuts, p, left=left, right=right)
                for p in pred  # iterate on individual prediction
            ]
        )
        assert r.shape == ((len(X), n_times) if n_times else (len(X),))
        return r

    def predict_survival(self, X, time):
        return self._interpolate_prediction(
            "predict_surv", X, time, left=1.0, right=0.0
        )

    def predict(self, X):
        return 1.0 - self.predict_survival(X, self.median_time_)

    @staticmethod
    def get_parameter_grid(max_width):
        return dict(
            layer_sizes=generate_topology_grid(max_width),
            # epochs=[100],
            # batch_size=[16, 32],
            dropout=[0.1, 0.2, 0.5],
            # validation_size=[0.1],
        )
