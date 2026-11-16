"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
``BasisState`` is the public basis-state form a run starts from.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Mapping, NamedTuple, TypeVar

import numpy as np

from ...registers import RegisterRef

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


@dataclass(frozen=True)
class BasisState:
    """A basis state given by one level per subsystem.

    Pass this object to ``run(initial_state=...)``, or as
    ``EvolutionState.quantum``, instead of a dense array. It names levels by
    reference, so it does not depend on subsystem order or, for a
    `fatqat.ModeRegister`, on the per-run cutoff. On a mode the level is the
    photon number. Every unlisted subsystem starts at level 0.

    Args:
        levels: Mapping from each `fatqat.RegisterRef` to its level, a
            non-negative int. A level must also be below the subsystem's
            dimension, which a run checks once the dimension is known.

    Raises:
        TypeError: If a key is not a `fatqat.RegisterRef` or a level is not an
            int.
        ValueError: If a level is negative.

    Examples:
        >>> import fatqat as fq
        >>> from fatqat.simulator import BasisState, Simulator
        >>> qubits = fq.QuantumRegister(2)
        >>> program = fq.Program([qubits])
        >>> state = (
        ...     Simulator("SV")
        ...     .run(program, initial_state=BasisState({qubits[1]: 1}))
        ...     .result()
        ...     .get_statevector()
        ... )
        >>> state.real
        array([0., 1., 0., 0.])
    """

    levels: Mapping[RegisterRef, int]

    def __post_init__(self) -> None:
        for ref, level in self.levels.items():
            if not isinstance(ref, RegisterRef):
                raise TypeError(
                    f"BasisState keys must be RegisterRef values, got {ref!r}"
                )
            if isinstance(level, (bool, np.bool_)) or not isinstance(
                level, (int, np.integer)
            ):
                raise TypeError(f"BasisState level for {ref} must be an int")
            if level < 0:
                raise ValueError(
                    f"BasisState level for {ref} must be non-negative, got {level}"
                )
        # A read-only copy keeps a caller's later edits out of the state.
        levels = MappingProxyType(
            {ref: int(level) for ref, level in self.levels.items()}
        )
        object.__setattr__(self, "levels", levels)


class _LoweredBasisState(NamedTuple):
    """A `BasisState` lowered to engine subsystems, for engine initialization.

    ``Simulator`` creates it from a `BasisState`: each reference resolves
    through the run's engine allocation, as gate targets do, so
    ``levels[i]`` is the level of engine subsystem ``i`` and engines never
    see references. It replaces the `BasisState` in
    ``EvolutionState.quantum``, the channel every execution path already
    carries to the engine, so no run signature changes.

    The engine builds the state from it in its own storage layout and array
    type, without an O(D) template: NumPy and Numba engines turn the levels
    into their flat index, and another array library or a non-dense engine
    can place them its own way. A plain tuple of levels would be
    indistinguishable from a dense ket, hence a distinct type.
    """

    levels: tuple[int, ...]


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
            the ground state; arrays and ``BasisState`` follow the rules of
            ``Simulator.run``.
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
