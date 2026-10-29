"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np

QuantumDataT = TypeVar("QuantumDataT")


@dataclass(slots=True)
class ClassicalState:
    """Classical registers, atom occupancy, and user data for one shot.

    Store user data in ``extra`` or add fields in a subclass.

    Attributes:
        clbits: Reported digits in classical-register order. None initializes
            all digits to zero. During custom execution this is a list of
            nonnegative int64 integers, one per classical-register entry.
        occupied: Present subsystem indices (least-significant tensor factor
            first). During execution, None means full occupancy. In a
            run's initial state, None requests the backend default: full for
            Simulator, empty for AtomArraySimulator. Changing this field alone
            does not reset quantum correlations.
        extra: Dictionary for user data; not included in counts.
    """

    # Unallocated register storage leaves unwritten report digits at zero.
    clbits: list[int] | None = None
    # None means implicitly full occupancy; an empty set means no carriers.
    occupied: set[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvolutionState(Generic[QuantumDataT]):
    """Quantum and classical data for one simulation shot.

    Pass this object to ``run(initial_state=...)`` to initialize both together.
    Each shot receives independent data, including a deep copy of the entire
    classical object. The supplied state is not modified.

    Attributes:
        quantum: During custom execution, a complex128 statevector or density
            matrix, according to the simulation method. Subsystem 0 is the
            least-significant tensor factor. For initialization, None selects
            the ground state; arrays follow the rules of ``Simulator.run``.
        classical: This shot's ``ClassicalState``.
    """

    quantum: QuantumDataT | None = None
    classical: ClassicalState = field(default_factory=ClassicalState)


class NumbaClassicalState(NamedTuple):
    """Classical report digits for a compiled shot; occupancy remains implicit."""

    clbits: np.ndarray


class NumbaEvolutionState(NamedTuple):
    """Quantum and classical storage owned by one compiled statevector shot."""

    quantum: np.ndarray
    classical: NumbaClassicalState
