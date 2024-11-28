from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _in_unstable_openblas_configuration  # noqa: F401
else:
    from sklearn.utils.fixes import _in_unstable_openblas_configuration  # noqa: F401
