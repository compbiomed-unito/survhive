"tests on bechmark datasets integrity and format conversions"

import survwrap as tosa
import pytest
import os
from sksurv.util import check_array_survival

_preloaded = {_name: tosa.get_data(_name) for _name in tosa.list_available_datasets()}
_shapes = {
    "flchain": (6524, 10),
    "gbsg2": (3668, 10),
    "metabric": (1903, 11),
    "support": (9105, 57),
}


def test_available_datasets():
    "check that that no available CSV is skipped"
    dataset_csvs = tuple(
        [
            _.split(".")[0]
            for _ in sorted(os.listdir(tosa.datasets._dataset_path))
            if _.endswith("csv")
        ]
    )
    assert tosa.list_available_datasets() == dataset_csvs


@pytest.mark.parametrize("a_dataset", tosa.list_available_datasets())
def test_no_nan_in_dataset(a_dataset):
    "test for no NaNs in dataset"
    assert not _preloaded[a_dataset].dataframe.isna().all().all()


@pytest.mark.parametrize("a_dataset", tosa.list_available_datasets())
def test_dataset_size(a_dataset):
    assert _preloaded[a_dataset].dataframe.to_numpy().shape == _shapes[a_dataset]


@pytest.mark.parametrize("a_dataset", tosa.list_available_datasets())
def test_get_X_y(a_dataset):
    Xt, yt = _preloaded[a_dataset].get_X_y()
    assert check_array_survival(Xt, yt)
