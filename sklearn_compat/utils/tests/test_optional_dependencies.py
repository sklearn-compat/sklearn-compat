import pytest

from sklearn_compat.utils._optional_dependencies import (
    check_matplotlib_support,
    check_pandas_support,
)


def test_check_matplotlib_support():
    is_matplotlib_installed = False
    try:
        import matplotlib  # noqa: F401

        is_matplotlib_installed = True
    except ImportError:
        pass

    if is_matplotlib_installed:
        check_matplotlib_support("sklearn_compat")
    else:
        with pytest.raises(ImportError):
            check_matplotlib_support("sklearn_compat")


def test_check_pandas_support():
    is_pandas_installed = False
    try:
        import pandas  # noqa: F401

        is_pandas_installed = True
    except ImportError:
        pass

    if is_pandas_installed:
        check_pandas_support("sklearn_compat")
    else:
        with pytest.raises(ImportError):
            check_pandas_support("sklearn_compat")
