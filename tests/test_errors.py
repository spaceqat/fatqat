"""Tests qnsim exception and warning hierarchy."""

from qnsim.errors import (
    QnsimError,
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
    NoMeasurementWarning,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, QnsimError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, QnsimError)
    assert issubclass(NoMeasurementWarning, UserWarning)
    assert issubclass(MatrixImplementationError, QnsimError)
