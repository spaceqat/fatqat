"""Tests fatqat exception and warning hierarchy."""

from fatqat.errors import (
    FatqcatError,
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
    NoMeasurementWarning,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, FatqcatError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, FatqcatError)
    assert issubclass(NoMeasurementWarning, UserWarning)
    assert issubclass(MatrixImplementationError, FatqcatError)
