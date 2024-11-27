from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _determine_key_type  # noqa: F401
    from sklearn.utils import _safe_indexing  # noqa: F401
    from sklearn.utils import _safe_assign  # noqa: F401
    from sklearn.utils import _get_column_indices  # noqa: F401
    from sklearn.utils import _get_column_indices_interchange  # noqa: F401
    from sklearn.utils import resample  # noqa: F401
    from sklearn.utils import shuffle  # noqa: F401

else:
    from sklearn.utils._indexing import _determine_key_type  # noqa: F401
    from sklearn.utils._indexing import _safe_indexing  # noqa: F401
    from sklearn.utils._indexing import _safe_assign  # noqa: F401
    from sklearn.utils._indexing import _get_column_indices  # noqa: F401
    from sklearn.utils._indexing import _get_column_indices_interchange  # noqa: F401
    from sklearn.utils._indexing import resample  # noqa: F401
    from sklearn.utils._indexing import shuffle  # noqa: F401
