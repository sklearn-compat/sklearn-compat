import numpy as np
from sklearn.base import (
    BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
)
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn_compat.base import ParamsValidationMixin, _fit_context
from sklearn_compat.utils.validation import validate_data
from sklearn_compat.utils._param_validation import Interval, Integral


class Classifier(ParamsValidationMixin, ClassifierMixin, BaseEstimator):

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
        return {
            "non_deterministic": True, "poor_score": True
        }


class Regressor(ParamsValidationMixin, RegressorMixin, BaseEstimator):

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
        return {
            "non_deterministic": True, "poor_score": True
        }


class Transformer(ParamsValidationMixin, TransformerMixin, BaseEstimator):

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


@parametrize_with_checks([Classifier(), Regressor(), Transformer()])
def test_basic_estimator(estimator, check):
    return check(estimator)
