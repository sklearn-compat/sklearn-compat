from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _get_column_indices # type: ignore[no-redef]
else:
    from sklearn.utils._indexing import _get_column_indices  # type: ignore[no-redef]
