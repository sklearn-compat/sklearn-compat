import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._testing import _convert_container

from sklearn_compat.utils.validation import (
    _check_feature_names,
    _check_n_features,
    _is_fitted,
    _to_object_array,
    check_array,
    check_X_y,
    validate_data,
)


def test_check_array_ensure_all_finite():
    X = [[1, 2, 3, np.nan]]
    with pytest.raises(ValueError, match="contains NaN"):
        check_array(X, ensure_all_finite=True)
    assert isinstance(check_array(X, ensure_all_finite=False), np.ndarray)


def test_check_X_y_ensure_all_finite():
    X, y = [[1, 2, 3, np.nan]], [1]
    with pytest.raises(ValueError, match="contains NaN"):
        check_X_y(X, y, ensure_all_finite=True)
    X_, y_ = check_X_y(X, y, ensure_all_finite=False)
    assert isinstance(X_, np.ndarray) and isinstance(y_, np.ndarray)


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


@pytest.mark.parametrize("container_type", ["list", "dataframe"])
def test_validate_data_skip_check_array(container_type):
    class MyEstimator(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            X = validate_data(self, X=X, y=y, skip_check_array=True)
            return self

        def transform(self, X):
            X = validate_data(self, X=X, skip_check_array=True)
            return X

    X = _convert_container(
        [[1, 2, 3, 4]], container_type, columns_name=["a", "b", "c", "d"]
    )
    est = MyEstimator()
    X_trans = est.fit_transform(X)
    assert isinstance(X_trans, type(X))
    assert est.n_features_in_ == 4
    if container_type == "dataframe":
        np.testing.assert_array_equal(est.feature_names_in_, ["a", "b", "c", "d"])


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
