import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_compat.utils.validation import validate_data

def test_validate_data():

    class MyEstimator(TransformerMixin,BaseEstimator):

        def fit(self, X, y=None):
            X = validate_data(self, X=X, y=y)
            return self

        def transform(self, X):
            X = validate_data(self, X=X)
            return X

    X = [[1, 2, 3, 4]]
    est = MyEstimator()
    assert isinstance(est.fit_transform(X), np.ndarray)
