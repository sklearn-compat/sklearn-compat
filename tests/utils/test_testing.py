"""Tests for the backward-compatible testing helpers."""

import numpy as np
import pytest

from sklearn_compat.utils._testing import _convert_container


def test_convert_container_array():
    """Check that `_convert_container` works for plain numpy arrays."""
    out = _convert_container([[1, 2, 3]], "array")
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("constructor_name", ["dataframe", "pandas"])
def test_convert_container_pandas_constructor_name(constructor_name):
    """Both the old ("dataframe") and new ("pandas") names should be accepted."""
    pd = pytest.importorskip("pandas")
    out = _convert_container([[1, 2, 3]], constructor_name)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize("kwarg_name", ["columns_name", "column_names"])
def test_convert_container_column_names(kwarg_name):
    """Both the old (`columns_name`) and new (`column_names`) keywords should work."""
    pytest.importorskip("pandas")
    out = _convert_container([[1, 2, 3]], "dataframe", **{kwarg_name: ["a", "b", "c"]})
    assert list(out.columns) == ["a", "b", "c"]


def test_convert_container_both_column_keywords_raises():
    """Passing both `columns_name` and `column_names` should raise."""
    pytest.importorskip("pandas")
    with pytest.raises(TypeError, match="not both"):
        _convert_container(
            [[1, 2, 3]],
            "dataframe",
            columns_name=["a", "b", "c"],
            column_names=["a", "b", "c"],
        )
