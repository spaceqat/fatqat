"""qnsim exception hierarchy and warnings."""

from __future__ import annotations


class QnsimError(Exception):
    """Base class for all qnsim errors."""


class BackendValidationError(QnsimError):
    """Raised at backend entry when a program/request is not acceptable."""


class UnsupportedOperationError(BackendValidationError):
    """Raised when the backend does not support an operation or feature."""


class MatrixImplementationError(QnsimError):
    """Raised when a registered implementation rule fails while building a
    plan's matrix step."""


class ResultFieldUnavailableError(QnsimError):
    """Raised when a Result field was not produced by this run."""


class NoMeasurementWarning(UserWarning):
    """Warned when counts include clbits that no measurement wrote."""
