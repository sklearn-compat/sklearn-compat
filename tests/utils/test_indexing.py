import numpy as np
import pytest

from sklearn_compat.utils._indexing import (
    _determine_key_type,
    _get_column_indices,
    _safe_indexing,
    resample,
    shuffle,
)


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
