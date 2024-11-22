import pytest
import numpy as np

from sklearn_compat.utils._indexing import _get_column_indices

def test_get_column_indices():
    """Check that we can import `_get_column_indices` from the right module.

    This change has been done in scikit-learn 1.5.
    """
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert np.array_equal(_get_column_indices(df, key="b"), [1])
