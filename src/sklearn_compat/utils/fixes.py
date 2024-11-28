import platform

from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _in_unstable_openblas_configuration  # noqa: F401
    from sklearn.utils import _IS_32BIT  # noqa: F401
    _IS_WASM = platform.machine() in ["wasm32", "wasm64"]
else:
    from sklearn.utils.fixes import _in_unstable_openblas_configuration  # noqa: F401
    from sklearn.utils.fixes import _IS_WASM  # noqa: F401
    from sklearn.utils.fixes import _IS_32BIT  # noqa: F401
