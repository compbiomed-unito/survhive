from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array

import pycox.models as Pycox
import numpy

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
    # qui mettiamo i parametri per la forma della rete, cercherei di fare qualcosa che rispetti il paper originale
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    epochs: int = 10 # maybe implement also early stopping
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = 'cpu'


    def fit(self, X, y):
        "fit a Pycox DeepHit model for single events"

        # from pycox.models import DeepHitSingle
        import torchtuples as tt
        X, y = check_X_y(X, y)
        optimizer = tt.optim.AdamWR(
            lr=self.learning_rate,
            #decoupled_weight_decay=,
            #cycle_eta_multiplier=,
        )
        self.labtrans_ = Pycox.DeepHitSingle.label_transform(self.num_durations)
        y_discrete = self.labtrans_.fit_transform(get_time(y), get_indicator(y))
        net = tt.practical.MLPVanilla(
            in_features=X.shape[1], 
            out_features=self.labtrans_.out_features, 
            num_nodes=self.layer_sizes,
            #batch_norm, dropout,
            #**self.model_params['indepnet'], **self.model_params['net']
        )
        self.model_ = Pycox.DeepHitSingle(
            net, optimizer, 
            device=self.device,
        )

        self.median_time_ = numpy.median(get_time(y))
        # BIG FAT WARNING: fit returns a TrainingLogger, not a fitted model. 
        # there are side effects on model_ itself
        self.model_.fit(
            X.astype('float32'), y_discrete, 
            num_workers=0 if True else n_jobs, 
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
        preds = 1 - self.model_.predict_surv(X.astype('float32'))
        #print('predict', eval_times.shape, self.labtrans_.cuts.shape, preds.shape)
        return numpy.array([numpy.interp(eval_times, self.labtrans_.cuts, p, left=0, right=1) for p in preds]).flatten()

    def concordance_index_censored(y_true, y_pred, *args, **kwargs):
        "return Harrel's C-index for a prediction"
        # print(y_true)
        from sksurv.metrics import concordance_index_censored
        return concordance_index_censored(
            event_indicator = get_indicator(y_true),
            event_time = get_time(y_true),
            estimate = y_pred,
            *args, **kwargs,
        )

    def score(self, X, y):
        "return the Harrel's c-index as a sklearn score"
        y_pred = self.model_.predict(X)
        return model_.concordance_index_censored(y, y_pred)[0]

#     def fit(self, X, y):
#         X, y = check_X_y(X, y)
#         self.model_.set_params(
#             l1_ratio=self.l1_ratio,
#             verbose=self.verbose,
#             fit_baseline_model=self.fit_baseline_model,
#         )
#         self.model_ = self.model_.fit(X, y)
#         return self

#     def predict(self, X):
#         X = check_array(X)
#         return self.model_.predict(X)

#     def score(self, X, y):
#         X, y = check_X_y(X, y)
#         return self.model_.score(X, y)

