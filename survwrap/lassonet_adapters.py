from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.utils import check_X_y, check_array
from sksurv.linear_models.coxph import BreslowEstimator

import numpy
import pandas
import torch
from os import getpid
from time import time

from .adapter import SurvivalEstimator
from .util import (
    get_time,
    get_indicator,
    survival_train_test_split,
)

_default_lambda_seq = [0.001 * 1.025**_ for _ in range(200)]

@dataclass
class FastCPH(SurvivalEstimator):
    """
    Adapter for the FastCPH method from lassonet

    NB: setting the parameter lambda_seq overrides the effects of both lambda_start and path_multiplier
    """

    package = "lassonet"
    model_ = SurvivalEstimator()

    # init
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    tie_approximation: str = "breslow"
    lambda_seq:  Sequence[float] = field( default_factory=lambda: _default_lambda_seq )
    lambda_start = (0.001,)
    path_multiplier = (1.025,)
    backtrack = (False,)
    device: str = None
    rng_seed: int = None
    verbose = bool = False
    fit_lambda_ = float

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed and self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def fit(self, X, y):
        from lassonet import LassoNetCoxRegressor

        # init
        X, y = check_X_y(X, y)

        self.model_ = LassoNetCoxRegressor(
            hidden_dims=tuple(self.layer_sizes),
            tie_approximation=self.tie_approximation,
            backtrack=self.backtrack,
            device=self.device,
        )
        if self.lambda_seq:
            self.model_.set_params(lambda_seq=self.lambda_seq)
        else:
            self.model_.set_params(
                lambda_start=self.lambda_start, path_multiplier=self.path_multiplier
            )

        if self.rng_seed:
            self._seed_rngs()
            self.model_.set_params(
                random_state=self.rng_seed,
                torch_seed=self.rng_seed,
            )
        if self.verbose:
            self.model_.set_params(verbose=2)
        else:
            self.model_.set_params(verbose=0)

        # fitting
        _y_lasso = numpy.column_stack(
            (get_time(y).astype("float32"), get_indicator(y).astype("float32"))
        )
        fastcph = self.model_.fit(X, _y_lasso)
        self.fit_lambda_ = min(fastcph.path_, key=(lambda x: x.objective)).lambda_
        # refitting on best lambda
        self.model_.set_params(lambda_seq=[self.fit_lambda_])
        _refit = self.model_.fit(X, _y_lasso)
        assert self.model_ == _refit
        return self

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X).flatten()

    # def predict(self, X, eval_times=None):
    #     X = check_array(X)
    #     if eval_times is None:
    #         eval_times = [self.median_time_]
    #     return 1 - self.predict_survival(X, eval_times).flatten()

    def _interpolate_prediction(self, method_name, X, time, left, right):
        X = check_array(X).astype("float32")
        try:
            n_times = len(time)
        except TypeError:
            n_times = 0
        pred = getattr(self.model_, method_name)(
            pandas.DataFrame(X.astype("float32"))
        ).cpu()
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

    @staticmethod
    def get_parameter_grid(max_width=None):
        return dict(
            hidden_factor=[4, 8],
            intermediate_size=[32, 64],
            num_hidden_layers=[2, 3, 4],
            num_attention_heads=[1, 2, 4],
        )
