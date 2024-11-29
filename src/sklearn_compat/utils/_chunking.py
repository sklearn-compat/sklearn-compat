from sklearn_compat.utils._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.5"):
    from sklearn.utils import _chunk_generator as chunk_generator  # noqa: F401
    from sklearn.utils import (
        gen_batches,  # noqa: F401
        gen_even_slices,  # noqa: F401
        get_chunk_n_rows,  # noqa: F401
    )
else:
    from sklearn.utils._chunking import (
        chunk_generator,  # noqa: F401
        gen_batches,  # noqa: F401
        gen_even_slices,  # noqa: F401
        get_chunk_n_rows,  # noqa: F401
    )
