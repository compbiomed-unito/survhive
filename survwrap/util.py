import numpy
import pandas as pd
from dataclasses import dataclass, field
from sksurv.datasets import get_x_y
from sklearn.utils import check_X_y
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold


def load_test_data(dataset="breast_cancer"):
    "Load standard breast-cancer dataset for testing"
    import sksurv.datasets
    from sklearn.preprocessing import OneHotEncoder

    # X, y = sksurv.datasets.load_breast_cancer()
    X, y = getattr(sksurv.datasets, "load_" + dataset)()
    X = numpy.concatenate(
        [
            X.select_dtypes("float"),
            OneHotEncoder(sparse_output=False).fit_transform(
                X.select_dtypes("category")
            ),
        ],
        axis=1,
    )
    X = SimpleImputer(strategy="median").fit_transform(X)

    return X, y


def get_indicator(y):
    "Get censoring indicator (bool)"
    return y[y.dtype.names[0]]


def get_time(y):
    "Get the time of the event"
    return y[y.dtype.names[1]]


## stratified partitioning


def survival_train_test_split(X, y, test_size=0.25, rng_seed=None, shuffle=True):
    "Split survival data into train and test set using event-label stratification"
    X, y = check_X_y(X, y)
    return train_test_split(
        X,
        y,
        stratify=get_indicator(y),
        test_size=test_size,
        random_state=rng_seed,
        shuffle=shuffle,
    )


def survival_crossval_splitter(X, y, n_splits=5, n_repeats=2, rng_seed=None):
    "a RepeatedStratifiedKFold CV splitter stratified according to survival events"
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=rng_seed
    ).split(X, get_indicator(y))


## Grid generation


def generate_topology_grid(max_width, max_layers=3):
    "return a list of net topologies to be used in hyperparameter optimization"
    from math import log

    start = 3
    base = 1.3
    topologies = []
    mono = [
        round(start * base ** (_))
        for _ in range(max_width)
        if _ < int((log(max_width) - log(start)) / log(base)) + 1
    ]
    for n_layers in range(1, max_layers + 1):
        topologies.extend([[_] * n_layers for _ in mono])
    return topologies


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
