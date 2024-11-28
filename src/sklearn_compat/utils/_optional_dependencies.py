from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import check_matplotlib_support  # noqa: F401
    from sklearn.utils import check_pandas_support  # noqa: F401
else:
    from sklearn.utils._optional_dependencies import (  # noqa: F401
        check_matplotlib_support,
    )
    from sklearn.utils._optional_dependencies import check_pandas_support  # noqa: F401
