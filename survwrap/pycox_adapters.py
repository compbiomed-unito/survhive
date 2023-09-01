from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.utils import check_X_y, check_array
from sksurv.metrics import concordance_index_censored

import pycox.models as Pycox
import numpy
import torch

from .adapter import SurvivalEstimator
from .util import get_time, get_indicator


@dataclass
class DeepHitSingle(SurvivalEstimator):
    """
    Adapter for the DeepHitSingle method from pycox
    """

    package = "pycox"
    model_ = SurvivalEstimator()
    verbose = False

    # init
    num_durations: int = 10
    # qui mettiamo i parametri per la forma della rete,
    # cercherei di fare qualcosa che rispetti il paper originale
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    epochs: int = 10  # maybe implement also early stopping
    batch_size: int = 16
    validation_size: float = 0.1
    learning_rate: float = 0.001
    device: str = "cpu"

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def fit(self, X, y):
        "fit a Pycox DeepHit model for single events"

        # from pycox.models import DeepHitSingle
        import torchtuples as tt

        self._seed_rngs()
        X, y = check_X_y(X, y)
        optimizer = tt.optim.AdamWR(
            lr=self.learning_rate,
            # decoupled_weight_decay=,
            # cycle_eta_multiplier=,
        )
        self.labtrans_ = Pycox.DeepHitSingle.label_transform(self.num_durations)
        y_discrete = self.labtrans_.fit_transform(get_time(y), get_indicator(y))
        net = tt.practical.MLPVanilla(
            in_features=X.shape[1],
            out_features=self.labtrans_.out_features,
            num_nodes=self.layer_sizes,
            # batch_norm, dropout,
            # **self.model_params['indepnet'], **self.model_params['net']
        )
        self.model_ = Pycox.DeepHitSingle(
            net,
            optimizer,
            device=self.device,
        )

        self.median_time_ = numpy.median(get_time(y))
        # BIG FAT WARNING: fit returns a TrainingLogger, not a fitted model.
        # there are side effects on model_ itself
        self.training_log_ = self.model_.fit(
            X.astype("float32"),
            y_discrete,
            # num_workers=0 if True else n_jobs,
            verbose=self.verbose,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        return self

    def predict(self, X, eval_times=None):
        "predict (1-S) with a Pycox DeepHit model for single events"
        X = check_array(X)
        if eval_times is None:
            eval_times = [self.median_time_]
        preds = 1 - self.model_.predict_surv(X.astype("float32"))
        # print('predict', eval_times.shape, self.labtrans_.cuts.shape, preds.shape)
        return numpy.array(
            [
                numpy.interp(eval_times, self.labtrans_.cuts, p, left=0, right=1)
                for p in preds
            ]
        ).flatten()

    def harrell_score(self, y_true, y_pred, *args, **kwargs):
        "return Harrell's C-index for a prediction"

        return concordance_index_censored(
            event_indicator=y_true[y_true.dtype.names[0]],
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
    def get_parameter_grid():
        return dict(
            num_durations=[10], layer_sizes=[[10, 10]], epochs=[10], batch_size=[16]
        )
