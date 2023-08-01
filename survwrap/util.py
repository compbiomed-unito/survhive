import numpy
import pandas as pd
from sksurv.datasets import get_x_y
from dataclasses import dataclass, field


def load_test_data():
    "Load standard breast-cancer dataset for testing"
    import sksurv.datasets
    from sklearn.preprocessing import OneHotEncoder

    X, y = sksurv.datasets.load_breast_cancer()
    X = numpy.concatenate(
        [
            X.select_dtypes("float"),
            OneHotEncoder(sparse_output=False).fit_transform(
                X.select_dtypes("category")
            ),
        ],
        axis=1,
    )
    return X, y


def get_indicator(y):
    "Get censoring indicator (bool)"
    return y[y.dtype.names[0]]


def get_time(y):
    "Get the time of the event"
    return y[y.dtype.names[1]]


## datasets


_dataset_path = "Datasets/"
_available_datasets = (
    "flchain",
    "gbsg2",
    "metabric",
    "support",
)


def list_available_datasets():
    "list the available benchmark datasets"
    return _available_datasets


def get_data(set_name):
    "Load one of the available benchmark datasets as a dataset object"
    if set_name not in _available_datasets:
        raise NameError("Dataset not available.")
    return dataset(
        name=set_name,
        dataframe=pd.read_csv(_dataset_path + set_name + ".csv", index_col=0),
    )


@dataclass
class dataset:
    "a dataset container"
    name: str
    dataframe: field(default_factory=pd.DataFrame)

    def get_X_y(self):
        "return dataset in scikit-survival format"
        return get_x_y(self.dataframe, attr_labels=["event", "time"], pos_label=1)
