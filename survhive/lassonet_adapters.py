from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Tuple, Union
from sklearn.utils import check_X_y, check_array
from sksurv.linear_model.coxph import BreslowEstimator
from lassonet import LassoNetCoxRegressor

import numpy
import torch

from .adapter import SurvivalEstimator
from .util import (
    get_time,
    get_indicator,
)

__all__ = [
    "FastCPH",
]

_default_lambda_seq = [0.001 * 1.025**_ for _ in range(200)]


@dataclass
class FastCPH(SurvivalEstimator):
    """
    Adapter for the FastCPH method from lassonet

    Notice:
        setting the parameter lambda_seq overrides the effects of
        BOTH lambda_start and path_multiplier
    """

    package = "lassonet"
    model_ = SurvivalEstimator()

    # init
    layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    tie_approximation: str = "efron"
    lambda_seq: Sequence[float] = field(default_factory=lambda: _default_lambda_seq)
    lambda_start: float = 0.001
    path_multiplier: float = 1.025
    backtrack: bool = False
    n_iters: Union[int, Tuple[int, int]] = (1000, 100)
    device: str = None
    rng_seed: int = None
    verbose: int = 0

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed and self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def _get_new_model(self, lambda_seq):
        return LassoNetCoxRegressor(
            hidden_dims=tuple(self.layer_sizes),
            tie_approximation=self.tie_approximation,
            backtrack=self.backtrack,
            device=self.device,
            verbose=self.verbose,
            lambda_seq=lambda_seq,
            lambda_start=self.lambda_start,
            path_multiplier=self.path_multiplier,
            n_iters=self.n_iters,
            random_state=self.rng_seed,
            torch_seed=self.rng_seed,
        )

    def fit(self, X, y):
        # init
        self.model_ = self._get_new_model(self.lambda_seq)
        if self.rng_seed is not None:
            self._seed_rngs()
        X, y = check_X_y(X, y)

        # fitting
        _y_times = get_time(y)
        _y_events = get_indicator(y)
        _y_lasso = numpy.column_stack(
            (_y_times.astype("float32"), _y_events.astype("float32"))
        )
        fastcph = self.model_.fit(X, _y_lasso)
        self.fit_lambda_ = min(fastcph.path_, key=(lambda x: x.objective)).lambda_

        # refitting on best lambda
        self.model_ = self._get_new_model(lambda_seq=[self.fit_lambda_])
        _refit = self.model_.fit(X, _y_lasso)
        assert self.model_ == _refit

        # fit breslow estimator too, to be used for time dependent predictions
        self.breslow_estimator_ = BreslowEstimator().fit(
            self.predict(X), _y_events, _y_times
        )

        return self

    def predict(self, X):
        X = check_array(X)
        return self.model_.predict(X).flatten()

    def predict_survival(self, X, time):
        X = check_array(X)
        try:
            n_times = len(time)
        except TypeError:
            n_times = 0
        pred = self.breslow_estimator_.get_survival_function(self.predict(X))
        r = numpy.array(
            [
                numpy.interp(time, p.x, p.y, left=1.0)
                for p in pred  # iterate on individual prediction
            ]
        )
        assert r.shape == ((len(X), n_times) if n_times else (len(X),))
        return r

    @staticmethod
    def get_parameter_grid(max_width=None):
        if max_width is not None:
            _mw = max_width
        else:
            _mw = 8
        return dict(
            layer_sizes=[[_mw], [_mw] * 2, [_mw] * 3, [_mw] * 4],
        )
