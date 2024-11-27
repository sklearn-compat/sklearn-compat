from sklearn_compat.utils._chunking import (
    chunk_generator,
    gen_batches,
    gen_even_slices,
    get_chunk_n_rows,
)


def test_chunk_generator():
    gen_chunk = chunk_generator(range(10), 3)
    assert len(next(gen_chunk)) == 3


def test_gen_batches():
    batches = list(gen_batches(7, 3))
    assert batches == [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]


def test_gen_even_slices():
    batches = list(gen_even_slices(10, 1))
    assert batches == [slice(0, 10, None)]


def test_get_chunk_n_rows():
    assert get_chunk_n_rows(10) == 107374182
