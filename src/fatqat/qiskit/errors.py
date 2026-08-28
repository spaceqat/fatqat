"""Qiskit integration errors."""

from __future__ import annotations

from ..errors import FatqatError


class QiskitConversionError(FatqatError):
    """A Qiskit circuit cannot be converted to a FATQAT program."""


class QiskitBackendError(FatqatError):
    """The FATQAT Qiskit backend rejected a run request before execution."""
