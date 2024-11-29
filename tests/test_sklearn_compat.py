import time

import numpy as np
import pytest
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import Integral, Interval, StrOptions
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn_compat._sklearn_compat import (
    _IS_32BIT,
    _IS_WASM,
    ParamsValidationMixin,
    _approximate_mode,
    _check_feature_names,
    _check_n_features,
    _construct_instances,
    _determine_key_type,
    _get_column_indices,
    _in_unstable_openblas_configuration,
    _is_fitted,
    _print_elapsed_time,
    _safe_indexing,
    _to_object_array,
    axis0_safe_slice,
    check_matplotlib_support,
    check_pandas_support,
    chunk_generator,
    gen_batches,
    gen_even_slices,
    get_chunk_n_rows,
    get_tags,
    indices_to_mask,
    is_pandas_na,
    is_scalar_nan,
    resample,
    safe_mask,
    safe_sqr,
    shuffle,
    validate_data,
)


class Classifier(ClassifierMixin, BaseEstimator):
    _parameter_constraints = {
        "seed": [Interval(Integral, 0, None, closed="left")],
    }
    _estimator_type = "classifier"

    def __init__(self, seed=0):
        self.seed = seed

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        X, y = validate_data(self, X=X, y=y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        indices = self.rng_.choice(len(self.classes_), size=X.shape[0], replace=True)
        return self.classes_[indices]

    def _more_tags(self):
        return {"non_deterministic": True, "poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.classifier_tags.poor_score = True
        return tags


class Regressor(RegressorMixin, BaseEstimator):
    _parameter_constraints = {
        "seed": [Interval(Integral, 0, None, closed="left")],
    }
    _estimator_type = "regressor"

    def __init__(self, seed=0):
        self.seed = seed

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        X, y = validate_data(self, X=X, y=y)
        self._min, self._max = y.min(), y.max()
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        return self.rng_.uniform(self._min, self._max, size=X.shape[0])

    def _more_tags(self):
        return {"non_deterministic": True, "poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.regressor_tags.poor_score = True
        return tags


class Transformer(TransformerMixin, BaseEstimator):
    _parameter_constraints = {"with_mean": ["boolean"], "with_std": ["boolean"]}
    _estimator_type = "transformer"

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = validate_data(self, X=X)
        self.mean_ = X.mean() if self.with_mean else 0
        self.std_ = X.std() if self.with_std else 1
        return self

    def transform(self, X):
        X = validate_data(self, X=X, reset=False)
        return (X - self.mean_) / self.std_


def test_chunk_generator():
    gen_chunk = chunk_generator(range(10), 3)
    assert len(next(gen_chunk)) == 3


def test_gen_batches():
    batches = list(gen_batches(7, 3))
    assert batches == [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]


def test_gen_even_slices():
    batches = list(gen_even_slices(10, 1))
    assert batches == [slice(0, 10, None)]


def test_get_chunk_n_rows():
    assert get_chunk_n_rows(10) == 107374182


def test__approximate_mode():
    result = _approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
    np.testing.assert_array_equal(result, np.array([2, 1]))


def test_safe_sqr():
    result = safe_sqr(np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([1, 4, 9]))


def test__in_unstable_openblas_configuration():
    _in_unstable_openblas_configuration()


def test__IS_WASM():
    assert not _IS_WASM


def test__IS_32BIT():
    assert not _IS_32BIT


def test__determine_key_type():
    assert _determine_key_type(np.arange(10)) == "int"


def test__safe_indexing():
    array = np.arange(10).reshape(2, 5)
    np.testing.assert_allclose(_safe_indexing(array, 1, axis=1), array[:, 1])


def test__get_column_indices():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert np.array_equal(_get_column_indices(df, key="b"), [1])


def test_resample():
    array = np.arange(10)
    resampled_array = resample(array, n_samples=20, replace=True, random_state=0)
    assert len(resampled_array) == 20


def test_shuffle():
    array = np.arange(10)
    shuffled_array = shuffle(array, random_state=0)
    assert len(shuffled_array) == len(array)


def test_safe_mask():
    data = np.arange(1, 6).reshape(-1, 1)
    condition = [False, True, True, False, True]
    mask = safe_mask(data, condition)
    np.testing.assert_array_equal(data[mask], np.array([[2], [3], [5]]))


def test_axis0_safe_slice():
    X = np.random.randn(5, 3)
    mask = np.array([True, False, True, False, True])
    result = axis0_safe_slice(X, mask, X.shape[0])
    np.testing.assert_array_equal(result, X[mask])


def test_indices_to_mask():
    indices = [1, 2, 3, 4]
    mask = indices_to_mask(indices, 5)
    np.testing.assert_array_equal(mask, np.array([False, True, True, True, True]))


def test_is_scalar_nan():
    assert is_scalar_nan(float("nan"))


def test_is_pandas_na():
    assert not is_pandas_na(float("nan"))


def test_check_matplotlib_support():
    is_matplotlib_installed = False
    try:
        import matplotlib  # noqa: F401

        is_matplotlib_installed = True
    except ImportError:
        pass

    if is_matplotlib_installed:
        check_matplotlib_support("sklearn_compat")
    else:
        with pytest.raises(ImportError):
            check_matplotlib_support("sklearn_compat")


def test_check_pandas_support():
    is_pandas_installed = False
    try:
        import pandas  # noqa: F401

        is_pandas_installed = True
    except ImportError:
        pass

    if is_pandas_installed:
        check_pandas_support("sklearn_compat")
    else:
        with pytest.raises(ImportError):
            check_pandas_support("sklearn_compat")


def test_get_tags():
    class MyEstimator(BaseEstimator):
        def _more_tags(self):
            return {"requires_fit": False}

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.requires_fit = False
            return tags

    tags = get_tags(MyEstimator())
    assert not tags.requires_fit


def test_print_elapsed_time():
    """Check that we can import `_print_elapsed_time` from the right module.

    This change has been done in scikit-learn 1.5.
    """
    import sys
    from io import StringIO

    stdout = StringIO()
    sys.stdout = stdout
    with _print_elapsed_time("sklearn_compat", "testing"):
        time.sleep(0.1)
    sys.stdout = sys.__stdout__
    output = stdout.getvalue()
    assert output.startswith("[sklearn_compat] .....")


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


@parametrize_with_checks(
    [
        Classifier(),
        Regressor(),
        Transformer(),
    ]
)
def test_basic_estimator(estimator, check):
    return check(estimator)


def test_parameter_validation():
    class TestEstimator(ParamsValidationMixin):
        """Estimator to which we apply parameter validation through the mixin."""

        _parameter_constraints = {
            "param": [StrOptions({"a", "b", "c"}), None],
        }

        def __init__(self, param=None):
            self.param = param

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            return self

        def get_params(self, deep=True):
            return {"param": self.param}

    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    TestEstimator().fit(X, y)
    TestEstimator(param="a").fit(X, y)
    with pytest.raises(ValueError, match="must be a str among"):
        TestEstimator(param="unknown").fit(X, y)


def test__construct_instances():
    list(iter(_construct_instances(LinearRegression)))
