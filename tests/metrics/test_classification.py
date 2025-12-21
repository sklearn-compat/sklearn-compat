from sklearn_compat.metrics._classification import _check_targets


def test__check_targets_returns_four_values():
    """Check that `_check_targets` always returns 4 values.

    This ensures compatibility across different sklearn versions by guaranteeing
    that the function always returns (y_type, y_true, y_pred, sample_weight).
    """
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    # Test without sample_weight
    result = _check_targets(y_true, y_pred)
    assert len(result) == 4
    y_type, y_true_out, y_pred_out, sample_weight_out = result
    assert sample_weight_out is None

    # Test with sample_weight
    sample_weight = [1.0, 2.0, 1.0, 2.0]
    result = _check_targets(y_true, y_pred, sample_weight=sample_weight)
    assert len(result) == 4
    y_type, y_true_out, y_pred_out, sample_weight_out = result
    assert sample_weight_out is not None
