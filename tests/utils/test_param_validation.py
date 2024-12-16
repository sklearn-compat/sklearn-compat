import pytest
from sklearn.utils._param_validation import InvalidParameterError

from sklearn_compat.utils._param_validation import validate_params


def test_validate_params():
    @validate_params({"x": [int, float]}, prefer_skip_nested_validation=True)
    def func(x):
        return x

    func(1)
    func(1.0)
    with pytest.raises(InvalidParameterError):
        func("a")
