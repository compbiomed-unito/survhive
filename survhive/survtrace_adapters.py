from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.utils import check_X_y, check_array

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

__all__ = [
    "SurvTraceSingle",
]


@dataclass
class SurvTraceSingle(SurvivalEstimator):
    """
    Adapter for the SurvTraceSingle method from SurvTRACE
    """

    package = "survtrace"
    model_ = SurvivalEstimator()

    # init
    num_durations: int = 5
    horizons: Sequence[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    #  'seed': 1234,
    # vocab_size: int = 8
    # the method 'hidden_size' parameter must be a multiple of num_attention_heads.
    # in order to enforce it during optimizations it is not exposed by the adapter
    # and internally computed as :
    # hidden_size = hidden_factor * num_attention_heads
    hidden_factor: int = 4
    # hidden_size: int = 16
    intermediate_size: int = 64
    num_hidden_layers: int = 3
    num_attention_heads: int = 2
    validation_size: float = 0.1
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.1
    patience: int = 5
    batch_size: int = 64
    epochs: int = 100
    rng_seed: int = None
    device: str = None

    # hidden_dropout_prob: float = 0.0
    #  'num_event': 1,
    #  'hidden_act': 'gelu',
    # attention_probs_dropout_prob: float = 0.1
    # early_stop_patience: 5
    # initializer_range: 0.001
    #  'layer_norm_eps': 1e-12,
    #  'max_position_embeddings': 512,
    #  'chunk_size_feed_forward': 0,
    #  'output_attentions': False,
    #  'output_hidden_states': False,
    #  'tie_word_embeddings': True,
    #  'pruned_heads': {}

    def _seed_rngs(self):
        "seed the random number generators involved in the model fit"
        if self.rng_seed and self.rng_seed > 0:
            numpy.random.seed(self.rng_seed)
            _ = torch.manual_seed(self.rng_seed)
            return True
        else:
            return False

    def fit(self, X, y):
        from survtrace.utils import LabelTransform
        from survtrace import STConfig, Trainer
        from survtrace.model import SurvTraceSingle

        X, y = check_X_y(X, y)
        self._seed_rngs()
        if self.device:
            torch.device(self.device)

        # detect competing risks
        num_risks = numpy.max(get_indicator(y).astype(float))
        assert num_risks == int(num_risks)
        num_risks = int(num_risks)

        # generate early-stopping train and validation set
        _X = {}
        _y = {}
        times = {}
        events = {}
        preprocessed_y = {}
        set_labels = ["train", "val"]
        _X["train"], _X["val"], _y["train"], _y["val"] = survival_train_test_split(
            X, y, test_size=self.validation_size, rng_seed=self.rng_seed
        )

        #
        for _ in set_labels:
            times[_] = get_time(_y[_])
            events[_] = get_indicator(_y[_])
            assert len(events[_].shape) == 1
        self.median_time_ = numpy.median(times["train"])

        # data preprocessing
        self.labtrans_ = LabelTransform(
            cuts=numpy.array(
                [times["train"].min()]
                + numpy.quantile(
                    times["train"][events["train"] == 1], STConfig["horizons"]
                ).tolist()
                + [times["train"].max()]
            )
        )
        self.labtrans_.fit(times["train"], events["train"])

        for _ in set_labels:
            preprocessed_y[_] = self.labtrans_.transform(times[_], events[_])

        preprocessed_y_df = {}
        for _ in set_labels:
            if num_risks == 1:
                preprocessed_y_df[_] = pandas.DataFrame(
                    {
                        "duration": preprocessed_y[_][0],
                        "event": preprocessed_y[_][1],
                        "proportion": preprocessed_y[_][2],
                    }
                )
            else:
                print("num_risks:", num_risks)
                preprocessed_y_df[_] = pandas.DataFrame(
                    {
                        "duration": preprocessed_y[_][0],
                        "proportion": preprocessed_y[_][2],
                    }
                )
                for evt in range(num_risks):
                    preprocessed_y_df[_]["event_" + str(evt)] = (
                        preprocessed_y[_][events[_] == (evt + 1)]
                    ).astype(float)

        # model setup
        # free parameters
        STConfig["num_durations"] = self.num_durations
        STConfig["horizons"] = self.horizons
        STConfig["hidden_size"] = self.hidden_factor * self.num_attention_heads
        STConfig["intermediate_size"] = self.intermediate_size
        STConfig["num_hidden_layers"] = self.num_hidden_layers
        STConfig["num_attention_heads"] = self.num_attention_heads
        STConfig["early_stop_patience"] = self.patience
        STConfig["hidden_dropout_prob"] = self.hidden_dropout
        STConfig["attention_probs_dropout_prob"] = self.attention_dropout

        # constrained parameters
        STConfig["seed"] = self.rng_seed
        STConfig["checkpoint"] = "./survtrace_checkpoints/" + "_".join(
            ["survtrace", str(time()), str(getpid())]
        )

        STConfig["labtrans"] = self.labtrans_
        STConfig["num_numerical_feature"] = X.shape[1]
        STConfig["num_categorical_feature"] = 0  # int(len(cols_categorical))
        STConfig["num_feature"] = X.shape[1]
        STConfig["vocab_size"] = 0
        STConfig["duration_index"] = self.labtrans_.cuts
        STConfig["out_feature"] = int(self.labtrans_.out_features)
        STConfig["num_event"] = num_risks

        self.model_ = SurvTraceSingle(STConfig)

        # initialize a trainer
        trainer = Trainer(self.model_)
        train_loss, val_loss = trainer.fit(
            (
                pandas.DataFrame(_X["train"].astype("float32")),
                preprocessed_y_df["train"],
            ),
            (pandas.DataFrame(_X["val"].astype("float32")), preprocessed_y_df["val"]),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        return self

    def predict(self, X, eval_times=None):
        X = check_array(X)
        if eval_times is None:
            eval_times = [self.median_time_]
        return 1 - self.predict_survival(X, eval_times).flatten()

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
