import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array,check_is_fitted
from sklearn.utils._param_validation import Interval, Integral

from sklearn_compat.utils.validation import validate_data


class MinimalClassifier:
    """Minimal classifier implementation without inheriting from BaseEstimator."""

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        self.classes_, counts = np.unique(y, return_counts=True)
        self._most_frequent_class_idx = counts.argmax()
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba_shape = (X.shape[0], self.classes_.size)
        y_proba = np.zeros(shape=proba_shape, dtype=np.float64)
        y_proba[:, self._most_frequent_class_idx] = 1.0
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = y_proba.argmax(axis=1)
        return self.classes_[y_pred]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


class MinimalRegressor:
    """Minimal regressor implementation without inheriting from BaseEstimator."""

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self._mean = np.mean(y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.ones(shape=(X.shape[0],)) * self._mean

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


class MinimalTransformer:
    """Minimal transformer implementation without inheriting from BaseEstimator."""

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has a different number of features than during fitting, "
                f"expected {self.n_features_in_}, got {X.shape[1]}."
            )
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


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
        MinimalClassifier(),
        MinimalRegressor(),
        MinimalTransformer(),
        Classifier(),
        Regressor(),
        Transformer(),
    ]
)
def test_basic_estimator(estimator, check):
    return check(estimator)
