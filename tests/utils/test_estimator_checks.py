from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_compat.utils.estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)
from sklearn_compat.utils.validation import validate_data


class MyEstimator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_data(self, X=X, y=y, reset=True)
        return self

    def transform(self, X):
        validate_data(self, X=X, reset=False)
        return X


failing_checks = {
    "check_transformer_data_not_an_array": "test",
    "check_transformers_unfitted": "test",
}


def test_check_estimator():
    check_estimator(
        MyEstimator(),
        expected_failed_checks=failing_checks,
    )


@parametrize_with_checks(
    [MyEstimator()], expected_failed_checks=lambda x: failing_checks
)
def test_parametrize_with_checks(estimator, check):
    check(estimator)
