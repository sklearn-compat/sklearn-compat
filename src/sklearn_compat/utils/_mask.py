from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import safe_mask  # noqa: F401
    from sklearn.utils import axis0_safe_slice  # noqa: F401
    from sklearn.utils import indices_to_mask  # noqa: F401
else:
    from sklearn.utils._mask import safe_mask  # noqa: F401
    from sklearn.utils._mask import axis0_safe_slice  # noqa: F401
    from sklearn.utils._mask import indices_to_mask  # noqa: F401
