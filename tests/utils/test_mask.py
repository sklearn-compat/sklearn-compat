import numpy as np

from sklearn_compat.utils._mask import axis0_safe_slice, indices_to_mask, safe_mask


def test_safe_mask():
    data = np.arange(1, 6).reshape(-1, 1)
    condition = [False, True, True, False, True]
    mask = safe_mask(data, condition)
    np.testing.assert_array_equal(data[mask], np.array([[2], [3], [5]]))


def test_axis0_safe_slice():
    X = np.random.randn(5, 3)
    mask = np.array([True, False, True, False, True])
    result = axis0_safe_slice(X, mask, X.shape[0])
    np.testing.assert_array_equal(result, X[mask])


def test_indices_to_mask():
    indices = [1, 2, 3, 4]
    mask = indices_to_mask(indices, 5)
    np.testing.assert_array_equal(mask, np.array([False, True, True, True, True]))
