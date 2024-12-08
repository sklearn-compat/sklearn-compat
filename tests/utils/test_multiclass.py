import pytest

from sklearn_compat.utils.multiclass import type_of_target


def type_of_target():
    with pytest.raises(ValueError):
        type_of_target([], raise_unknown=True)
