"Test run of SurvTRACE on METABRIC dataset"

# import pdb
from numpy.testing import assert_array_almost_equal


from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.train_utils import Trainer
from survtrace.config import STConfig

test_rounding = 2

# define the setup parameters
STConfig["data"] = "metabric"
# STConfig['seed'] = 2309

set_random_seed(STConfig["seed"])

hparams = {
    "batch_size": 64,
    "weight_decay": 1e-4,
    "learning_rate": 1e-3,
    "epochs": 20,
}


# load data
df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)

# get model
model = SurvTraceSingle(STConfig)

# initialize a trainer
trainer = Trainer(model)
train_loss, val_loss = trainer.fit(
    (df_train, df_y_train),
    (df_val, df_y_val),
    batch_size=hparams["batch_size"],
    epochs=hparams["epochs"],
    learning_rate=hparams["learning_rate"],
    weight_decay=hparams["weight_decay"],
)


# evaluate model
evaluator = Evaluator(df, df_train.index)


def test_survtrace_metabric():
    eval = evaluator.eval(model, (df_test, df_y_test))
    assert_array_almost_equal(
        [eval["0.25_brier"], eval["0.5_brier"], eval["0.75_brier"]],
        [0.110, 0.185, 0.210],
        decimal=test_rounding,
    )
    assert_array_almost_equal(
        [eval["0.25_ipcw"], eval["0.5_ipcw"], eval["0.75_ipcw"]],
        [0.720, 0.697, 0.683],
        decimal=test_rounding,
    )
