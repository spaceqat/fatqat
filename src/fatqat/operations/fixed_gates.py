"""Fixed (parameter-free) unitary gates: single-qubit and multi-qubit.

Examples:
    Build a Bell pair with ``H`` then ``CX``:

    >>> import fatqat as fq
    >>> import fatqat.operations as ops
    >>> program = fq.Program(2)
    >>> program.add(ops.H, 0)
    >>> program.add(ops.CX, (0, 1))
    >>> result = fq.simulator.Simulator("SV").run(
    ...     program,
    ...     result_config={"counts": False, "final_state": True},
    ... ).result()
    >>> result.get_statevector()
    array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation

# ---------------------------------------------------------------------------
# Fixed single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HGate(Operation):
    """Put one qubit into or out of an equal superposition.

    In ``|0>, |1>`` basis order, the matrix is
    ``[[1, 1], [1, -1]] / sqrt(2)``. Use the singleton ``ops.H`` without
    parentheses.
    """

    name: ClassVar[str] = "H"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class IGate(Operation):
    """Leave one qubit unchanged.

    The matrix is ``[[1, 0], [0, 1]]``. Use the singleton ``ops.I`` without
    parentheses.
    """

    name: ClassVar[str] = "I"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SGate(Operation):
    """Apply the S phase gate, the square root of Z, to one qubit.

    The matrix is ``diag(1, i)``. Use the singleton ``ops.S`` without
    parentheses.
    """

    name: ClassVar[str] = "S"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SdgGate(Operation):
    """Apply the inverse S phase gate to one qubit.

    The matrix is ``diag(1, -i)``. Use the singleton ``ops.Sdg`` without
    parentheses.
    """

    name: ClassVar[str] = "Sdg"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SXGate(Operation):
    """Apply the principal square root of X to one qubit.

    The matrix is ``[[1+i, 1-i], [1-i, 1+i]] / 2``. Use the singleton
    ``ops.SX`` without parentheses.
    """

    name: ClassVar[str] = "SX"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    """Apply a pi/4 phase to the ``|1>`` amplitude of one qubit.

    The matrix is ``diag(1, exp(i*pi/4))``. Use the singleton ``ops.T``
    without parentheses.
    """

    name: ClassVar[str] = "T"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TdgGate(Operation):
    """Apply a -pi/4 phase to the ``|1>`` amplitude of one qubit.

    The matrix is ``diag(1, exp(-i*pi/4))``. Use the singleton ``ops.Tdg``
    without parentheses.
    """

    name: ClassVar[str] = "Tdg"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    """Exchange ``|0>`` and ``|1>`` on one qubit.

    The Pauli-X matrix is ``[[0, 1], [1, 0]]``. Use the singleton ``ops.X``
    without parentheses.
    """

    name: ClassVar[str] = "X"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    """Apply the Pauli-Y bit-and-phase flip to one qubit.

    The matrix is ``[[0, -i], [i, 0]]``. Use the singleton ``ops.Y`` without
    parentheses.
    """

    name: ClassVar[str] = "Y"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    """Negate the ``|1>`` amplitude of one qubit.

    The Pauli-Z matrix is ``diag(1, -1)``. Use the singleton ``ops.Z`` without
    parentheses.
    """

    name: ClassVar[str] = "Z"
    num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Fixed multi-qubit unitary gates (2+ qubits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CXGate(Operation):
    """Flip a target qubit when its control is ``|1>``.

    Targets are ``(control, target)``. ``CX`` also accepts a compatible pair
    of :class:`~fatqat.RegisterView` values and applies the gate member by
    member. Use the singleton ``ops.CX`` without parentheses.
    """

    name: ClassVar[str] = "CX"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CZGate(Operation):
    """Negate ``|11>`` and leave the other two-qubit basis states unchanged.

    Targets are ``(control, target)``. ``CZ`` also accepts a compatible pair
    of :class:`~fatqat.RegisterView` values and applies the gate member by
    member. Use the singleton ``ops.CZ`` without parentheses.
    """

    name: ClassVar[str] = "CZ"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SwapGate(Operation):
    """Exchange the states of two qubits.

    Use the singleton ``ops.Swap`` without parentheses on two scalar targets.
    """

    name: ClassVar[str] = "Swap"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CYGate(Operation):
    """Apply Pauli-Y to a target qubit when its control is ``|1>``.

    Targets are ``(control, target)``. Use the singleton ``ops.CY`` without
    parentheses.
    """

    name: ClassVar[str] = "CY"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CSGate(Operation):
    """Apply S to a target qubit when its control is ``|1>``.

    Targets are ``(control, target)`` and the matrix is ``diag(1, 1, 1, i)``.
    Use the singleton ``ops.CS`` without parentheses.
    """

    name: ClassVar[str] = "CS"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class iSwapGate(Operation):
    """Swap ``|01>`` and ``|10>`` while multiplying each by ``i``.

    Use the singleton ``ops.iSwap`` without parentheses on two scalar targets.
    """

    name: ClassVar[str] = "iSwap"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CCXGate(Operation):
    """Flip a target qubit when both controls are ``|1>``.

    The Toffoli target order is ``(control0, control1, target)``. Use the
    singleton ``ops.CCX`` without parentheses.
    """

    name: ClassVar[str] = "CCX"
    num_subsystems: ClassVar[int] = 3


@dataclass(frozen=True)
class CSwapGate(Operation):
    """Exchange two target qubits when the control is ``|1>``.

    The Fredkin target order is ``(control, target0, target1)``. Use the
    singleton ``ops.CSwap`` without parentheses.
    """

    name: ClassVar[str] = "CSwap"
    num_subsystems: ClassVar[int] = 3


# ---------------------------------------------------------------------------
# Public fixed-gate instances
# ---------------------------------------------------------------------------
# These classes have no parameters, so each is exported only as a singleton
# value (e.g. `ops.H`), not as a class - unlike parametric gates, there is
# no reason for a caller to ever name `HGate` itself.

H = HGate()
I = IGate()
S = SGate()
Sdg = SdgGate()
SX = SXGate()
T = TGate()
Tdg = TdgGate()
X = XGate()
Y = YGate()
Z = ZGate()
CX = CXGate()
CZ = CZGate()
Swap = SwapGate()
CY = CYGate()
CS = CSGate()
iSwap = iSwapGate()
CCX = CCXGate()
CSwap = CSwapGate()
