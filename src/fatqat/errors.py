"""fatqat exception hierarchy."""

from __future__ import annotations


class FatqatError(Exception):
    """Base class for FATQAT-defined errors."""


class BackendValidationError(FatqatError):
    """A backend cannot run the supplied program or request."""


class BackendExecutionError(FatqatError):
    """A backend failed after accepting an execution request.

    A backend run may store this error on a failed `Job`; calling
    `job.result()` then raises it. APIs that return a value directly, such as
    pulse-emulator propagator construction, raise it directly.
    """


class UnsupportedOperationError(BackendValidationError):
    """A backend does not support an operation or requested feature."""


class MatrixImplementationError(FatqatError):
    """A registered matrix rule raised while resolving an operation."""


class PulseImplementationError(FatqatError):
    """A pulse rule raised or did not return a `PulseDefinition`.

    `BackendValidationError`, including `UnsupportedOperationError`, passes
    through unchanged when the rule raises it deliberately.
    """


class ResultFieldUnavailableError(FatqatError):
    """The requested `Result` field was not produced by the run."""
