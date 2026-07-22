"""fatqat exception hierarchy and warnings."""

from __future__ import annotations


class FatqcatError(Exception):
    """Base class for all fatqat errors."""


class BackendValidationError(FatqcatError):
    """Raised at backend entry when a program/request is not acceptable."""


class UnsupportedOperationError(BackendValidationError):
    """Raised when the backend does not support an operation or feature."""


class UnsupportedResourceOperandError(BackendValidationError):
    """Raised when no resource binder can resolve a frontend target expression."""


class MatrixImplementationError(FatqcatError):
    """Raised when a registered implementation rule fails while building a
    plan's matrix step."""


class ResultFieldUnavailableError(FatqcatError):
    """Raised when a Result field was not produced by this run."""


class NoMeasurementWarning(UserWarning):
    """Warned when counts include clbits that no measurement wrote."""
