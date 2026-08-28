"""Optional Qiskit conversion and ``BackendV2`` integration.

Import this namespace only when Qiskit is installed. Use
:func:`circuit_to_program` for one-way circuit conversion, or construct a
:class:`FatqatBackend` for Qiskit's transpiler and backend-based primitives.
"""

from __future__ import annotations

from .backend import FatqatBackend
from .converter import circuit_to_program
from .errors import QiskitBackendError, QiskitConversionError
from .job import FatqatJob
from .provider import FatqatProvider
from .target import build_simulator_target

__all__ = [
    "FatqatBackend",
    "FatqatJob",
    "FatqatProvider",
    "QiskitBackendError",
    "QiskitConversionError",
    "build_simulator_target",
    "circuit_to_program",
]
