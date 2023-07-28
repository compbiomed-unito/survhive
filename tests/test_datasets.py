"tests on bechmark datasets integrity and format conversions"

import survwrap as tosa
import pytest
import os


def get_shape(a_dataset):
    "return the known shape of sthe full dataset _dataframe_"
    _shapes = {
        "flchain": (6524, 10),
        "gbsg2": (3668, 10),
        "metabric": (1903, 11),
        "support": (9105, 57),
    }
    return _shapes[a_dataset]


def test_available_datasets():
    "check that that no available CSV is skipped"
    dataset_path = "Datasets"
    dataset_csvs = tuple([_.split(".")[0] for _ in sorted(os.listdir(dataset_path))])
    assert tosa.list_available_datasets() == dataset_csvs


@pytest.mark.parametrize("a_dataset", tosa.list_available_datasets())
def test_no_nan_in_dataset(a_dataset):
    "test for no NaNs in dataset"
    assert not tosa.get_data(a_dataset).dataframe.isna().all().all()


@pytest.mark.parametrize("a_dataset", tosa.list_available_datasets())
def test_dataset_size(a_dataset):
    assert tosa.get_data(a_dataset).dataframe.to_numpy().shape == get_shape(a_dataset)
