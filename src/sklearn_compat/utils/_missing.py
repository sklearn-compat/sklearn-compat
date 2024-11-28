from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import is_scalar_nan  # noqa: F401
    from sklearn.utils import _is_pandas_na as is_pandas_na  # noqa: F401
else:
    from sklearn.utils._missing import is_scalar_nan  # noqa: F401
    from sklearn.utils._missing import is_pandas_na  # noqa: F401
