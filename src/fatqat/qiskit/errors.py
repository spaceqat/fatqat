"""Qiskit integration errors."""

from __future__ import annotations

from ..errors import FatqatError


class QiskitConversionError(FatqatError):
    """Raised when a Qiskit circuit cannot be converted to a fatqat Program."""


class QiskitBackendError(FatqatError):
    """Raised when the Qiskit backend rejects a run request."""
