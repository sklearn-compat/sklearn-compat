from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _print_elapsed_time  # noqa: F401
else:
    from sklearn.utils._user_interface import _print_elapsed_time  # noqa: F401
