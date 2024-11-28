import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_compat.utils.validation import (
    validate_data,
    _check_n_features,
    _check_feature_names,
    _is_fitted,
    _to_object_array,
)


@pytest.mark.parametrize("ensure_all_finite", [True, False])
def test_validate_data(ensure_all_finite):
    """Check the behaviour of `validate_data`.

    This change has been introduced in scikit-learn 1.6.
    """

    class MyEstimator(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            X = validate_data(self, X=X, y=y, ensure_all_finite=ensure_all_finite)
            return self

        def transform(self, X):
            X = validate_data(self, X=X, ensure_all_finite=ensure_all_finite)
            return X

    X = [[1, 2, 3, 4]]
    est = MyEstimator()
    assert isinstance(est.fit_transform(X), np.ndarray)

    X = [[1, 2, 3, np.nan]]
    est = MyEstimator()
    if ensure_all_finite:
        with pytest.raises(ValueError, match="contains NaN"):
            est.fit_transform(X)
    else:
        assert isinstance(est.fit_transform(X), np.ndarray)


def test_check_n_features():
    """Check the behaviour of `_check_n_features`.

    This change has been introduced in scikit-learn 1.6.
    """

    class MyEstimator(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            X = _check_n_features(self, X, reset=True)
            return self

    X = [[1, 2, 3, 4]]
    est = MyEstimator().fit(X)
    assert est.n_features_in_ == 4


def test_check_feature_names():
    """Check the behaviour of `_check_feature_names`.

    This change has been introduced in scikit-learn 1.6.
    """
    pd = pytest.importorskip("pandas")

    class MyEstimator(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            X = _check_feature_names(self, X, reset=True)
            return self

    X = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    est = MyEstimator().fit(X)
    np.testing.assert_array_equal(est.feature_names_in_, ["a", "b", "c", "d"])


def test__to_object_array():
    result = _to_object_array([np.array([0]), np.array([1])])
    assert isinstance(result, np.ndarray)
    assert result.dtype == object


def test__is_fitted():
    estimator = BaseEstimator()
    assert not _is_fitted(estimator)
