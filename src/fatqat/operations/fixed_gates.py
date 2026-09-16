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
    """Apply the Hadamard transform to one qubit.

    In ``|0>, |1>`` basis order, the matrix is
    ``[[1, 1], [1, -1]] / sqrt(2)``.
    """

    name: ClassVar[str] = "H"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class HYGate(Operation):
    """Apply the LQCloud-native ``Y H`` single-qubit operation."""

    name: ClassVar[str] = "HY"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class IGate(Operation):
    """Leave one qubit unchanged.

    The matrix is ``[[1, 0], [0, 1]]``.
    """

    name: ClassVar[str] = "I"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SGate(Operation):
    """Apply the S phase gate, the square root of Z, to one qubit.

    The matrix is ``diag(1, i)``.
    """

    name: ClassVar[str] = "S"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SdgGate(Operation):
    """Apply the inverse S phase gate to one qubit.

    The matrix is ``diag(1, -i)``.
    """

    name: ClassVar[str] = "Sdg"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SXGate(Operation):
    """Apply the principal square root of X to one qubit.

    The matrix is ``[[1+i, 1-i], [1-i, 1+i]] / 2``.
    """

    name: ClassVar[str] = "SX"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class TGate(Operation):
    """Apply a pi/4 phase to the ``|1>`` amplitude of one qubit.

    The matrix is ``diag(1, exp(i*pi/4))``.
    """

    name: ClassVar[str] = "T"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class TdgGate(Operation):
    """Apply a -pi/4 phase to the ``|1>`` amplitude of one qubit.

    The matrix is ``diag(1, exp(-i*pi/4))``.
    """

    name: ClassVar[str] = "Tdg"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class XGate(Operation):
    """Exchange ``|0>`` and ``|1>`` on one qubit.

    The Pauli-X matrix is ``[[0, 1], [1, 0]]``.
    """

    name: ClassVar[str] = "X"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MXGate(Operation):
    """Apply minus Pauli-X, the negative of the X matrix."""

    name: ClassVar[str] = "MX"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class YGate(Operation):
    """Apply the Pauli-Y bit-and-phase flip to one qubit.

    The matrix is ``[[0, -i], [i, 0]]``.
    """

    name: ClassVar[str] = "Y"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MYGate(Operation):
    """Apply minus Pauli-Y, the negative of the Y matrix."""

    name: ClassVar[str] = "MY"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class ZGate(Operation):
    """Negate the ``|1>`` amplitude of one qubit.

    The Pauli-Z matrix is ``diag(1, -1)``.
    """

    name: ClassVar[str] = "Z"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MZGate(Operation):
    """Apply minus Pauli-Z, equivalent to ``RZ(-pi)`` up to phase."""

    name: ClassVar[str] = "MZ"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class XHalfGate(Operation):
    """Rotate by ``pi/2`` about the positive X axis."""

    name: ClassVar[str] = "XHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MXHalfGate(Operation):
    """Rotate by ``pi/2`` about the negative X axis."""

    name: ClassVar[str] = "MXHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class YHalfGate(Operation):
    """Rotate by ``pi/2`` about the positive Y axis."""

    name: ClassVar[str] = "YHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MYHalfGate(Operation):
    """Rotate by ``pi/2`` about the negative Y axis."""

    name: ClassVar[str] = "MYHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class XYHalfGate(Operation):
    """Rotate by ``pi/2`` about ``(X + Y) / sqrt(2)``."""

    name: ClassVar[str] = "XYHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MXYHalfGate(Operation):
    """Rotate by ``pi/2`` about ``(-X + Y) / sqrt(2)``."""

    name: ClassVar[str] = "MXYHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class MXMYHalfGate(Operation):
    """Rotate by ``pi/2`` about ``(-X - Y) / sqrt(2)``."""

    name: ClassVar[str] = "MXMYHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class XMYHalfGate(Operation):
    """Rotate by ``pi/2`` about ``(X - Y) / sqrt(2)``."""

    name: ClassVar[str] = "XMYHalf"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Fixed multi-qubit unitary gates (2+ qubits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CXGate(Operation):
    """Flip a target qubit when its control is ``|1>``.

    Targets are ``(control, target)``.
    """

    name: ClassVar[str] = "CX"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CZGate(Operation):
    """Negate ``|11>`` and leave the other two-qubit basis states unchanged.

    Targets are ``(control, target)``.
    """

    name: ClassVar[str] = "CZ"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SwapGate(Operation):
    """Exchange the states of two qubits.

    Targets are two distinct qubits.
    """

    name: ClassVar[str] = "Swap"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CYGate(Operation):
    """Apply Pauli-Y to a target qubit when its control is ``|1>``.

    Targets are ``(control, target)``.
    """

    name: ClassVar[str] = "CY"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CSGate(Operation):
    """Apply S to a target qubit when its control is ``|1>``.

    Targets are ``(control, target)`` and the matrix is ``diag(1, 1, 1, i)``.
    """

    name: ClassVar[str] = "CS"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class iSwapGate(Operation):
    """Swap ``|01>`` and ``|10>`` while multiplying each by ``i``.

    Targets are two distinct qubits.
    """

    name: ClassVar[str] = "iSwap"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CCXGate(Operation):
    """Flip a target qubit when both controls are ``|1>``.

    The Toffoli target order is ``(control0, control1, target)``.
    """

    name: ClassVar[str] = "CCX"
    num_subsystems: ClassVar[int] = 3
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class CSwapGate(Operation):
    """Exchange two target qubits when the control is ``|1>``.

    The Fredkin target order is ``(control, target0, target1)``.
    """

    name: ClassVar[str] = "CSwap"
    num_subsystems: ClassVar[int] = 3
    _accepts_views: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Public fixed-gate instances
# ---------------------------------------------------------------------------
H = HGate()
HY = HYGate()
I = IGate()
S = SGate()
Sdg = SdgGate()
SX = SXGate()
T = TGate()
Tdg = TdgGate()
X = XGate()
MX = MXGate()
Y = YGate()
MY = MYGate()
Z = ZGate()
MZ = MZGate()
XHalf = XHalfGate()
MXHalf = MXHalfGate()
YHalf = YHalfGate()
MYHalf = MYHalfGate()
XYHalf = XYHalfGate()
MXYHalf = MXYHalfGate()
MXMYHalf = MXMYHalfGate()
XMYHalf = XMYHalfGate()
CX = CXGate()
CZ = CZGate()
Swap = SwapGate()
CY = CYGate()
CS = CSGate()
iSwap = iSwapGate()
CCX = CCXGate()
CSwap = CSwapGate()
