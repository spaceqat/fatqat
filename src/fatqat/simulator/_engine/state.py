"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
"""

from dataclasses import dataclass, field
from typing import Generic, NamedTuple, TypeVar

import numpy as np

QuantumDataT = TypeVar("QuantumDataT")


@dataclass(slots=True)
class QuantumState(Generic[QuantumDataT]):
    """The authoritative quantum data and its concrete representation.

    The engine determines the data type and layout. This record does not
    prescribe numerical operations or imply support for arbitrary arrays.
    """

    data: QuantumDataT
    representation: str


@dataclass(slots=True)
class ClassicalState:
    """Classical container whose components are allocated only when needed.

    Components may stay on the CPU. Updates based on numerical results wait for
    those results and finish before later operations use them.
    """

    # Unallocated register storage leaves unwritten report digits at zero.
    clbits: list[int] | None = None
    # None means implicitly full occupancy; an empty set means no carriers.
    occupied: set[int] | None = None


@dataclass(slots=True)
class EvolutionState(Generic[QuantumDataT]):
    """Quantum and classical containers owned by one active evolution.

    An empty classical container allocates no component buffers and does not
    indicate capability support. Every evolution receives its own container.
    """

    quantum: QuantumState[QuantumDataT]
    classical: ClassicalState = field(default_factory=ClassicalState)


class NumbaQuantumState(NamedTuple):
    """Quantum data for a compiled statevector shot."""

    data: np.ndarray


class NumbaClassicalState(NamedTuple):
    """Classical report digits for a compiled shot; occupancy remains implicit."""

    clbits: np.ndarray


class NumbaEvolutionState(NamedTuple):
    """Quantum and classical storage owned by one compiled statevector shot."""

    quantum: NumbaQuantumState
    classical: NumbaClassicalState
