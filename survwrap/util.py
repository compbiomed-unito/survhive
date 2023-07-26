import numpy
import sksurv.datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_test_data(dataset='breast_cancer'):
    "Load standard dataset for testing"

    #X, y = sksurv.datasets.load_breast_cancer()
    X, y = getattr(sksurv.datasets, 'load_' + dataset)()
    X = numpy.concatenate(
        [
            X.select_dtypes("float"),
            OneHotEncoder(sparse_output=False).fit_transform(
                X.select_dtypes("category")
            ),
        ],
        axis=1,
    )
    X = SimpleImputer(strategy='median').fit_transform(X)

    return X, y


def get_indicator(y):
    "Get censoring indicator (bool)"
    return y[y.dtype.names[0]]


def get_time(y):
    "Get the time of the event"
    return y[y.dtype.names[1]]
