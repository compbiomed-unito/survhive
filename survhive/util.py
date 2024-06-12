import numpy
from sklearn.utils import check_X_y
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
)

__all__ = [
    "load_test_data",
    "get_indicator",
    "get_time",
    "event_quantiles",
    "survival_train_test_split",
    "survival_crossval_splitter",
]


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


def event_quantiles(y, quantiles=[0.25, 0.5, 0.75]):
    "get the times corresponding to the specified quantile fractions of events"
    return numpy.quantile(get_time(y)[get_indicator(y)], quantiles)


## stratified partitioning


def survival_train_test_split(X, y, test_size=0.25, rng_seed=None, shuffle=True):
    "Split survival data into train and test set using event-label stratification"
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
