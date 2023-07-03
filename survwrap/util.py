
import numpy as np

def load_test_data():
    "Load standard dataset for testing"
    import sksurv.datasets
    from sklearn.preprocessing import OneHotEncoder

    X, y = sksurv.datasets.load_breast_cancer()
    X = np.concatenate(
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

