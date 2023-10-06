from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from sklearn.utils import check_X_y, check_array

import numpy
import pandas

from .adapter import SurvivalEstimator
from .util import (
    get_time,
    get_indicator,
)


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
    # qui mettiamo i parametri per la forma della rete,
    # cercherei di fare qualcosa che rispetti il paper originale
    # layer_sizes: Sequence[int] = field(default_factory=lambda: [10, 10])
    # vocab_size: int = 8
    hidden_size: int = 16
    intermediate_size: int = 64
    num_hidden_layers: int = 3
    num_attention_heads: int = 2
    rng_seed: int = None
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

    # def fit(self, X, times, events):
    def fit(self, X, y_):
        from survtrace.utils import LabelTransform
        from survtrace import STConfig, Trainer
        from survtrace.model import SurvTraceSingle

        X, y_ = check_X_y(X, y_)
        times = get_time(y_)
        events = get_indicator(y_)
        self.median_time_ = numpy.median(times)

        #
        assert len(events.shape) == 1
        num_risks = numpy.max(events.astype(float))
        assert num_risks == int(num_risks)
        num_risks = int(num_risks)

        # data preprocessing
        self.labtrans_ = LabelTransform(
            cuts=numpy.array(
                [times.min()]
                + numpy.quantile(times[events == 1], STConfig["horizons"]).tolist()
                + [times.max()]
            )
        )
        self.labtrans_.fit(times, events)
        y = self.labtrans_.transform(times, events)

        if num_risks == 1:
            ydf = pandas.DataFrame(
                {"duration": y[0], "event": y[1], "proportion": y[2]}
            )
        else:
            print("num_risks:", num_risks)
            ydf = pandas.DataFrame({"duration": y[0], "proportion": y[2]})
            for evt in range(num_risks):
                ydf["event_" + str(evt)] = (events == (evt + 1)).astype(float)
            print(
                pandas.concat(
                    [ydf, pandas.DataFrame({"events": events, "y[1]": y[1]})], axis=1
                ).head(20)
            )

        # free parameters
        STConfig["num_durations"] = self.num_durations
        STConfig["horizons"] = self.horizons
        STConfig["hidden_size"] = self.hidden_size
        STConfig["intermediate_size"] = self.intermediate_size
        STConfig["num_hidden_layers"] = self.num_hidden_layers
        STConfig["num_attention_heads"] = self.num_attention_heads

        # constrained parameters
        STConfig["seed"] = self.rng_seed

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
            (pandas.DataFrame(X.astype("float32")), ydf),
        )
        return self

    def predict(self, X, eval_times=None):
        X = check_array(X)
        if eval_times is None:
            eval_times = [self.median_time_]
        preds = 1 - self.model_.predict_surv(pandas.DataFrame(X.astype("float32")))
        return numpy.array(
            [
                numpy.interp(eval_times, self.labtrans_.cuts, p, left=0, right=1)
                for p in preds.cpu()
            ]
        ).flatten()


# class SurvTraceMethod(SurvMethod):
#     short_name = "survtrace"
#     long_name = "SurvTRACE"
#     multi = False
#     competing = False  # TODO add competing!

#     @staticmethod
#     def trial2model_params(trial):
#         # Got the error: "The hidden size (16) is not a 
          # multiple of the number of attention heads (6)"
#         # So the hidden size is now a multiple of the attention head number
#         num_attention_heads = trial.suggest_int("num_attention_heads", 2, 6)
#         return dict(
#             conf=dict(
#                 hidden_size=num_attention_heads
#                 * trial.suggest_int("hidden_size_log2", 1, 6),
#                 intermediate_size=2
#                 ** trial.suggest_int("intermediate_size_log2", 3, 7),
#                 hidden_dropout_proba=trial.suggest_float(
#                     "hidden_dropout_proba", 1e-3, 0.5, log=True
#                 ),
#                 num_hidden_layers=trial.suggest_int("num_hidden_layers", 1, 4),
#                 num_attention_heads=num_attention_heads,
#             ),
#             fit=dict(
#                 learning_rate=trial.suggest_float(
#                     "learning_rate", 1e-4, 1e-2, log=True
#                 ),
#                 epochs=2 ** trial.suggest_int("epochs_log2", 2, 6),
#                 batch_size=2 ** trial.suggest_int("batch_size_log2", 3, 8),
#                 weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-2,log=True),
#             ),
#         )

#     # STConfig defaults:
#     #  'num_durations': 5,
#     #  'horizons': [0.25, 0.5, 0.75],
#     #  'seed': 1234,
#     #  'checkpoint': './checkpoints/survtrace.pt',
#     #  'vocab_size': 8,
#     #  'hidden_size': 16,
#     #  'intermediate_size': 64,
#     #  'num_hidden_layers': 3,
#     #  'num_attention_heads': 2,
#     #  'hidden_dropout_prob': 0.0,
#     #  'num_feature': 9,
#     #  'num_numerical_feature': 5,
#     #  'num_categorical_feature': 4,
#     #  'out_feature': 3,
#     #  'num_event': 1,
#     #  'hidden_act': 'gelu',
#     #  'attention_probs_dropout_prob': 0.1,
#     #  'early_stop_patience': 5,
#     #  'initializer_range': 0.001,
#     #  'layer_norm_eps': 1e-12,
#     #  'max_position_embeddings': 512,
#     #  'chunk_size_feed_forward': 0,
#     #  'output_attentions': False,
#     #  'output_hidden_states': False,
#     #  'tie_word_embeddings': True,
#     #  'pruned_heads': {}

#     def fit(self, X, times, events):
#         from survtrace.utils import LabelTransform
#         from survtrace import STConfig, Trainer
#         from survtrace.model import SurvTraceSingle

#         assert len(events.shape) == 1
#         num_risks = numpy.max(events.astype(float))
#         assert num_risks == int(num_risks)
#         num_risks = int(num_risks)

#         # data preprocessing
#         self.labtrans_ = LabelTransform(
#             cuts=numpy.array(
#                 [times.min()]
#                 + numpy.quantile(times[events == 1], STConfig["horizons"]).tolist()
#                 + [times.max()]
#             )
#         )
#         self.labtrans_.fit(times, events)
#         y = self.labtrans_.transform(times, events)

#         if num_risks == 1:
#             ydf = pandas.DataFrame(
#                 {"duration": y[0], "event": y[1], "proportion": y[2]}
#             )
#         else:
#             print("num_risks:", num_risks)
#             ydf = pandas.DataFrame({"duration": y[0], "proportion": y[2]})
#             for evt in range(num_risks):
#                 ydf["event_" + str(evt)] = (events == (evt + 1)).astype(float)
#             print(
#                 pandas.concat(
#                     [ydf, pandas.DataFrame({"events": events, "y[1]": y[1]})], axis=1
#                 ).head(20)
#             )

#         STConfig["labtrans"] = self.labtrans_
#         STConfig["num_numerical_feature"] = X.shape[1]
#         STConfig["num_categorical_feature"] = 0  # int(len(cols_categorical))
#         STConfig["num_feature"] = X.shape[1]
#         STConfig["vocab_size"] = 0
#         STConfig["duration_index"] = self.labtrans_.cuts
#         STConfig["out_feature"] = int(self.labtrans_.out_features)

#         STConfig["num_event"] = num_risks

#         STConfig.update(self.model_params["conf"])
#         self.model_ = SurvTraceSingle(STConfig)

#         # initialize a trainer
#         trainer = Trainer(self.model_)
#         train_loss, val_loss = trainer.fit(
#             (pandas.DataFrame(X.astype("float32")), ydf), **self.model_params["fit"]
#         )
#         # global dbg
#         # dbg = self, X, y
#         return self

#     def predict(self, X, eval_times):
#         preds = 1 - self.model_.predict_surv(pandas.DataFrame(X.astype("float32")))
#         return numpy.array(
#             [
#                 numpy.interp(eval_times, self.labtrans_.cuts, p, left=0, right=1)
#                 for p in preds.cpu()
#             ]
#         )
