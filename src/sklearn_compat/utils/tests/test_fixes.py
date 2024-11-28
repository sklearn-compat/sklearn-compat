from sklearn_compat.utils.fixes import (
    _in_unstable_openblas_configuration,
    _IS_WASM,
    _IS_32BIT,
)


def test__in_unstable_openblas_configuration():
    _in_unstable_openblas_configuration()


def test__IS_WASM():
    assert not _IS_WASM

def test__IS_32BIT():
    assert not _IS_32BIT
