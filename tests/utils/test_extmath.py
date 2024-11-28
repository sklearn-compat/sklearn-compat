import numpy as np

from sklearn_compat.utils.extmath import safe_sqr, _approximate_mode


def test__approximate_mode():
    result = _approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
    np.testing.assert_array_equal(result, np.array([2, 1]))


def test_safe_sqr():
    result = safe_sqr(np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([1, 4, 9]))
