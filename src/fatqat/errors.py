"""fatqat exception hierarchy."""

from __future__ import annotations


class FatqatError(Exception):
    """Base class for all fatqat errors."""


class BackendValidationError(FatqatError):
    """Raised at backend entry when a program/request is not acceptable."""


class BackendExecutionError(FatqatError):
    """Raised by :meth:`~fatqat.Job.result` when backend execution fails."""


class UnsupportedOperationError(BackendValidationError):
    """Raised when the backend does not support an operation or feature."""


class MatrixImplementationError(FatqatError):
    """Raised when a registered implementation rule fails while building a
    plan's matrix step."""


class PulseImplementationError(FatqatError):
    """Raised when a registered pulse implementation rule fails, or returns
    something other than a :class:`~fatqat.emulator.PulseDefinition`, while
    lowering a pulse program occurrence. A rule's own
    :exc:`BackendValidationError` (including
    :exc:`UnsupportedOperationError`) propagates unchanged instead - that is
    the rule's deliberate validation, not an implementation defect."""


class ResultFieldUnavailableError(FatqatError):
    """Raised when a Result field was not produced by this run."""
