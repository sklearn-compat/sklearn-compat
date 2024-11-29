from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import (
        axis0_safe_slice,  # noqa: F401
        indices_to_mask,  # noqa: F401
        safe_mask,  # noqa: F401
    )
else:
    from sklearn.utils._mask import (
        axis0_safe_slice,  # noqa: F401
        indices_to_mask,  # noqa: F401
        safe_mask,  # noqa: F401
    )
