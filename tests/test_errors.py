"""Tests the fatqat exception hierarchy."""

from fatqat.errors import (
    FatqatError,
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, FatqatError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, FatqatError)
    assert issubclass(MatrixImplementationError, FatqatError)
