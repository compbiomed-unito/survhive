"Some frequently used benchmark datasets for survival analysis"


import pandas as pd
import pkgutil
from dataclasses import dataclass, field
from sksurv.datasets import get_x_y
from os import path
from sys import modules


_dataset_path = path.dirname(modules[__name__].__file__) + '/datasets/'
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
        dataframe=pd.read_csv( _dataset_path + set_name + ".csv",
            index_col=0),
    )


@dataclass
class dataset:
    "a dataset container"
    name: str
    dataframe: field(default_factory=pd.DataFrame)

    def get_X_y(self):
        "return dataset in scikit-survival format"
        return get_x_y(self.dataframe, attr_labels=["event", "time"], pos_label=1)