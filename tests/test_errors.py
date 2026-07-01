"""Tests qnsim exception and warning hierarchy."""

from qnsim.errors import (
    QnsimError,
    BackendValidationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
    NoMeasurementWarning,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, QnsimError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, QnsimError)
    assert issubclass(NoMeasurementWarning, UserWarning)
