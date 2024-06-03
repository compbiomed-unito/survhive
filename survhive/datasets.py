"Some frequently used benchmark datasets for survival analysis"


from numpy import ndarray
import pandas as pd
from os import path
from sys import modules
from dataclasses import dataclass, field
from sksurv.datasets import get_x_y

__all__ = [
    "list_available_datasets",
    "get_data",
    "dataset",
]

_dataset_path = path.dirname(modules[__name__].__file__) + "/datasets/"
_available_datasets = (
    "flchain",
    "gbsg2",
    "metabric",
    "support",
)


def list_available_datasets():
    "list of the available benchmark datasets"
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
        # "return dataset in scikit-survival format"
        """Returns the dataset in scikit-survival format).

        Returns:
            two numpy ndarrays containing features and (event,time) ndarray.
        """

        _X, _y = get_x_y(self.dataframe, attr_labels=["event", "time"], pos_label=1)
        return _X.to_numpy(dtype=ndarray), _y

    def index_zero_times(self):
        """
        Returns pandas indexes of event with a zero time.
        Usually these data points should be removed.
        Removal can be performed **inplace** using the *discard_zero_times* method.
        """
        return self.dataframe[self.dataframe["time"] == 0].index

    def discard_zero_times(self):
        """
        In-place discards (*side effects!*) data points with zero times.

        Returns:
            the new shape of the dataset.
        """
        _zeroes = self.index_zero_times()
        if not _zeroes.empty:
            self.dataframe.drop(_zeroes, inplace=True)
        return self.dataframe.shape
