"""qnsim exception hierarchy and warnings."""

from __future__ import annotations


class QnsimError(Exception):
    """Base class for all qnsim errors."""


class BackendValidationError(QnsimError):
    """Raised at backend entry when a program/request is not acceptable."""


class UnsupportedOperationError(BackendValidationError):
    """Raised when the backend does not support an operation or feature."""


class ResultFieldUnavailableError(QnsimError):
    """Raised when a Result field was not produced by this run."""


class NoMeasurementWarning(UserWarning):
    """Warned when counts contain only never-written clbits and no state is delivered."""
