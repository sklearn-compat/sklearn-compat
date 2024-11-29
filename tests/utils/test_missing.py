from sklearn_compat.utils._missing import is_pandas_na, is_scalar_nan


def test_is_scalar_nan():
    assert is_scalar_nan(float("nan"))


def test_is_pandas_na():
    assert not is_pandas_na(float("nan"))
