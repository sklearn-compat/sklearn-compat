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
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, Integral, StrOptions

from sklearn_compat.base import ParamsValidationMixin
from sklearn_compat.utils.validation import validate_data

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
