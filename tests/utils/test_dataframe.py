"""Tests for dataframe detection functions."""

import numpy as np
import pytest

from sklearn._min_dependencies import dependent_packages
from sklearn.utils._testing import _convert_container

from sklearn_compat._sklearn_compat import parse_version, sklearn_version
from sklearn_compat.utils._dataframe import (
    is_df_or_series,
    is_pandas_df,
    is_pandas_df_or_series,
    is_polars_df,
    is_polars_df_or_series,
    is_pyarrow_data,
)


@pytest.mark.parametrize(
    "constructor_name",
    [
        pytest.param(
            "pyarrow",
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages,
                reason="pyarrow not in dependent_packages",
            ),
        ),
        "dataframe",
        pytest.param(
            "polars",
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages,
                reason="polars not in dependent_packages",
            ),
        ),
    ],
)
def test_is_df_or_series(constructor_name):
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)

    assert is_df_or_series(df)
    assert not is_df_or_series(np.asarray([1, 2, 3]))


@pytest.mark.parametrize(
    "constructor_name",
    [
        pytest.param(
            "pyarrow",
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages,
                reason="pyarrow not in dependent_packages",
            ),
        ),
        "dataframe",
        pytest.param(
            "polars",
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages,
                reason="polars not in dependent_packages",
            ),
        ),
    ],
)
def test_is_pandas_df_other_libraries(constructor_name):
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    if constructor_name in ("pyarrow", "polars"):
        assert not is_pandas_df(df)
    else:
        assert is_pandas_df(df)


def test_is_pandas_df():
    """Check behavior of is_pandas_df when pandas is installed."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3]])
    assert is_pandas_df(df)
    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


def test_is_pandas_df_pandas_not_installed(hide_available_pandas):
    """Check is_pandas_df when pandas is not installed."""

    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        pytest.param(
            "pyarrow",
            dependent_packages.get("pyarrow", [None])[0],
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="pyarrow not in dependent_packages or sklearn < 1.4",
            ),
        ),
        ("dataframe", dependent_packages["pandas"][0]),
        pytest.param(
            "polars",
            dependent_packages.get("polars", [None])[0],
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="polars not in dependent_packages or sklearn < 1.4",
            ),
        ),
    ],
)
def test_is_polars_df_other_libraries(constructor_name, minversion):
    if constructor_name == "dataframe":
        df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    else:
        df = _convert_container(
            [[1, 4, 2], [3, 3, 6]],
            constructor_name,
            minversion=minversion,
        )
    if constructor_name in ("pyarrow", "dataframe"):
        assert not is_polars_df(df)
    else:
        assert is_polars_df(df)


def test_is_polars_df_for_duck_typed_polars_dataframe():
    """Check is_polars_df for object that looks like a polars dataframe"""

    class NotAPolarsDataFrame:
        def __init__(self):
            self.columns = [1, 2, 3]
            self.schema = "my_schema"

    not_a_polars_df = NotAPolarsDataFrame()
    assert not is_polars_df(not_a_polars_df)


def test_is_polars_df():
    """Check that is_polars_df return False for non-dataframe objects."""

    class LooksLikePolars:
        def __init__(self):
            self.columns = ["a", "b"]
            self.schema = ["a", "b"]

    assert not is_polars_df(LooksLikePolars())


def test_is_pandas_df_or_series():
    """Check behavior of is_pandas_df_or_series when pandas is installed."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3]])
    series = pd.Series([1, 2, 3])
    assert is_pandas_df_or_series(df)
    assert is_pandas_df_or_series(series)
    assert not is_pandas_df_or_series(np.asarray([1, 2, 3]))
    assert not is_pandas_df_or_series(1)


def test_is_pandas_df_or_series_pandas_not_installed(hide_available_pandas):
    """Check is_pandas_df_or_series when pandas is not installed."""
    assert not is_pandas_df_or_series(np.asarray([1, 2, 3]))
    assert not is_pandas_df_or_series(1)


@pytest.mark.parametrize(
    "constructor_name",
    [
        pytest.param(
            "pyarrow",
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages,
                reason="pyarrow not in dependent_packages",
            ),
        ),
        "dataframe",
        pytest.param(
            "polars",
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages,
                reason="polars not in dependent_packages",
            ),
        ),
    ],
)
def test_is_pandas_df_or_series_other_libraries(constructor_name):
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    if constructor_name in ("pyarrow", "polars"):
        assert not is_pandas_df_or_series(df)
    else:
        assert is_pandas_df_or_series(df)


def test_is_polars_df_or_series():
    """Check behavior of is_polars_df_or_series when polars is installed."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    series = pl.Series("a", [1, 2, 3])
    assert is_polars_df_or_series(df)
    assert is_polars_df_or_series(series)
    assert not is_polars_df_or_series(np.asarray([1, 2, 3]))
    assert not is_polars_df_or_series(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        pytest.param(
            "pyarrow",
            dependent_packages.get("pyarrow", [None])[0],
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="pyarrow not in dependent_packages or sklearn < 1.4",
            ),
        ),
        ("dataframe", dependent_packages["pandas"][0]),
        pytest.param(
            "polars",
            dependent_packages.get("polars", [None])[0],
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="polars not in dependent_packages or sklearn < 1.4",
            ),
        ),
    ],
)
def test_is_polars_df_or_series_other_libraries(constructor_name, minversion):
    if constructor_name == "dataframe":
        df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    else:
        df = _convert_container(
            [[1, 4, 2], [3, 3, 6]],
            constructor_name,
            minversion=minversion,
        )
    if constructor_name in ("pyarrow", "dataframe"):
        assert not is_polars_df_or_series(df)
    else:
        assert is_polars_df_or_series(df)


def test_is_pyarrow_data():
    """Check behavior of is_pyarrow_data when pyarrow is installed."""
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    record_batch = pa.record_batch({"a": [1, 2, 3], "b": [4, 5, 6]})
    array = pa.array([1, 2, 3])
    chunked_array = pa.chunked_array([[1, 2, 3], [4, 5, 6]])
    assert is_pyarrow_data(table)
    assert is_pyarrow_data(record_batch)
    assert is_pyarrow_data(array)
    assert is_pyarrow_data(chunked_array)
    assert not is_pyarrow_data(np.asarray([1, 2, 3]))
    assert not is_pyarrow_data(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        pytest.param(
            "pyarrow",
            dependent_packages.get("pyarrow", [None])[0],
            marks=pytest.mark.skipif(
                "pyarrow" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="pyarrow not in dependent_packages or sklearn < 1.4",
            ),
        ),
        ("dataframe", dependent_packages["pandas"][0]),
        pytest.param(
            "polars",
            dependent_packages.get("polars", [None])[0],
            marks=pytest.mark.skipif(
                "polars" not in dependent_packages
                or sklearn_version < parse_version("1.4"),
                reason="polars not in dependent_packages or sklearn < 1.4",
            ),
        ),
    ],
)
def test_is_pyarrow_data_other_libraries(constructor_name, minversion):
    if constructor_name == "dataframe":
        df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    else:
        df = _convert_container(
            [[1, 4, 2], [3, 3, 6]],
            constructor_name,
            minversion=minversion,
        )
    if constructor_name == "pyarrow":
        assert is_pyarrow_data(df)
    else:
        assert not is_pyarrow_data(df)
