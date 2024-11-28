from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import safe_sqr  # noqa: F401
    from sklearn.utils import _approximate_mode  # noqa: F401
else:
    from sklearn.utils.extmath import safe_sqr  # noqa: F401
    from sklearn.utils.extmath import _approximate_mode  # noqa: F401
