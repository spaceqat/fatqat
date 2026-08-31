"""Qiskit integration errors."""

from __future__ import annotations

from qiskit.exceptions import QiskitError

from ..errors import FatqatError


class QiskitConversionError(FatqatError, QiskitError):
    """A Qiskit circuit cannot be converted to a FATQAT program.

    Also a ``qiskit.exceptions.QiskitError`` so one ``except QiskitError:``
    covers both the adapter's direct raises and errors re-raised from a
    failed job, mirroring how ``QASMTranspileError`` stays a ``ValueError``.
    """


class QiskitBackendError(FatqatError, QiskitError):
    """The FATQAT Qiskit backend rejected a run request before execution.

    Also a ``qiskit.exceptions.QiskitError``; see `QiskitConversionError`.
    """
