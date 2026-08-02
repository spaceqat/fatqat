"""Tests fatqat exception and warning hierarchy."""

from fatqat.errors import (
    FatqatError,
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
    NoMeasurementWarning,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, FatqatError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, FatqatError)
    assert issubclass(NoMeasurementWarning, UserWarning)
    assert issubclass(MatrixImplementationError, FatqatError)
