from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import (
        _determine_key_type,  # noqa: F401
        _get_column_indices,  # noqa: F401
        _safe_assign,  # noqa: F401
        _safe_indexing,  # noqa: F401
        resample,  # noqa: F401
        shuffle,  # noqa: F401
    )

else:
    from sklearn.utils._indexing import (
        _determine_key_type,  # noqa: F401
        _get_column_indices,  # noqa: F401
        _safe_assign,  # noqa: F401
        _safe_indexing,  # noqa: F401
        resample,  # noqa: F401
        shuffle,  # noqa: F401
    )
