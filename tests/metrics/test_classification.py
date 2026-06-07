import numpy as np

from sklearn_compat.metrics._classification import _check_targets


def test__check_targets_returns_five_values():
    """Check that `_check_targets` always returns 5 values.

    This ensures compatibility across different sklearn versions by guaranteeing
    that the function always returns
    (y_type, unique_labels, y_true, y_pred, sample_weight). The `unique_labels`
    element was introduced in scikit-learn 1.9.
    """
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    # Test without sample_weight
    result = _check_targets(y_true, y_pred)
    assert len(result) == 5
    y_type, labels, y_true_out, y_pred_out, sample_weight_out = result
    assert y_type == "binary"
    np.testing.assert_array_equal(labels, [0, 1])
    assert sample_weight_out is None

    # Test with sample_weight
    sample_weight = [1.0, 2.0, 1.0, 2.0]
    result = _check_targets(y_true, y_pred, sample_weight=sample_weight)
    assert len(result) == 5
    y_type, labels, y_true_out, y_pred_out, sample_weight_out = result
    np.testing.assert_array_equal(labels, [0, 1])
    assert sample_weight_out is not None
