"""Fixed (parameter-free) unitary gates: single-qubit and multi-qubit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation

# ---------------------------------------------------------------------------
# Fixed single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HGate(Operation):
    """Hadamard gate operation.

    .. math::

        H = \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}
    """

    name: ClassVar[str] = "H"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class IGate(Operation):
    """Identity gate operation.

    .. math::

        I = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}
    """

    name: ClassVar[str] = "I"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SGate(Operation):
    """S phase gate operation (square root of Z).

    .. math::

        S = \\begin{pmatrix} 1 & 0 \\\\ 0 & i \\end{pmatrix}
    """

    name: ClassVar[str] = "S"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SdgGate(Operation):
    """Inverse S phase gate operation.

    .. math::

        S^\\dagger = \\begin{pmatrix} 1 & 0 \\\\ 0 & -i \\end{pmatrix}
    """

    name: ClassVar[str] = "Sdg"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    """T phase gate operation.

    .. math::

        T = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\pi/4} \\end{pmatrix}
    """

    name: ClassVar[str] = "T"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TdgGate(Operation):
    """Inverse T phase gate operation.

    .. math::

        T^\\dagger = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{-i\\pi/4} \\end{pmatrix}
    """

    name: ClassVar[str] = "Tdg"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    """Pauli-X gate operation.

    .. math::

        X = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}
    """

    name: ClassVar[str] = "X"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    """Pauli-Y gate operation.

    .. math::

        Y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}
    """

    name: ClassVar[str] = "Y"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    """Pauli-Z gate operation.

    .. math::

        Z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}
    """

    name: ClassVar[str] = "Z"
    _num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Fixed multi-qubit unitary gates (2+ qubits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CXGate(Operation):
    """Controlled-X gate operation.

    ``targets = (control, target)``; operand 0 is the control. Basis order
    :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        CX = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&0&1\\\\ 0&0&1&0
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CX"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CZGate(Operation):
    """Controlled-Z gate operation.

    ``targets = (control, target)``; operand 0 is the control. Basis order
    :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        CZ = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&1&0\\\\ 0&0&0&-1
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CZ"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class SwapGate(Operation):
    """Swap gate operation: exchanges the state of its two targets.

    Basis order :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        \\mathrm{Swap} = \\begin{pmatrix}
        1&0&0&0\\\\ 0&0&1&0\\\\ 0&1&0&0\\\\ 0&0&0&1
        \\end{pmatrix}
    """

    name: ClassVar[str] = "Swap"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CYGate(Operation):
    """Controlled-Y gate operation.

    ``targets = (control, target)``; operand 0 is the control. Basis order
    :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        CY = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&0&-i\\\\ 0&0&i&0
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CY"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CSGate(Operation):
    """Controlled-S gate operation.

    ``targets = (control, target)``; operand 0 is the control. Basis order
    :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        CS = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&1&0\\\\ 0&0&0&i
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CS"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class iSwapGate(Operation):
    """iSWAP gate operation: swaps its two targets, applying an i phase to
    the swapped amplitudes.

    Basis order :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        i\\mathrm{Swap} = \\begin{pmatrix}
        1&0&0&0\\\\ 0&0&i&0\\\\ 0&i&0&0\\\\ 0&0&0&1
        \\end{pmatrix}
    """

    name: ClassVar[str] = "iSwap"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CCXGate(Operation):
    """Doubly-controlled-X (Toffoli) gate operation.

    ``targets = (control0, control1, target)``; operands 0 and 1 are the
    controls. Basis order :math:`|000\\rangle, \\dots, |111\\rangle`; flips
    the target iff both controls are :math:`|1\\rangle`.

    .. math::

        CCX = \\begin{pmatrix}
        1&0&0&0&0&0&0&0\\\\
        0&1&0&0&0&0&0&0\\\\
        0&0&1&0&0&0&0&0\\\\
        0&0&0&1&0&0&0&0\\\\
        0&0&0&0&1&0&0&0\\\\
        0&0&0&0&0&1&0&0\\\\
        0&0&0&0&0&0&0&1\\\\
        0&0&0&0&0&0&1&0
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CCX"
    _num_subsystems: ClassVar[int] = 3


@dataclass(frozen=True)
class CSwapGate(Operation):
    """Controlled-swap (Fredkin) gate operation.

    ``targets = (control, target0, target1)``; operand 0 is the control.
    Basis order :math:`|000\\rangle, \\dots, |111\\rangle`; exchanges the
    two targets iff the control is :math:`|1\\rangle`.

    .. math::

        CSwap = \\begin{pmatrix}
        1&0&0&0&0&0&0&0\\\\
        0&1&0&0&0&0&0&0\\\\
        0&0&1&0&0&0&0&0\\\\
        0&0&0&1&0&0&0&0\\\\
        0&0&0&0&1&0&0&0\\\\
        0&0&0&0&0&0&1&0\\\\
        0&0&0&0&0&1&0&0\\\\
        0&0&0&0&0&0&0&1
        \\end{pmatrix}
    """

    name: ClassVar[str] = "CSwap"
    _num_subsystems: ClassVar[int] = 3


# ---------------------------------------------------------------------------
# Public fixed-gate instances
# ---------------------------------------------------------------------------
# These classes have no parameters, so each is exported only as a singleton
# value (e.g. `fqc.ops.H`), not as a class - unlike parametric gates, there is
# no reason for a caller to ever name `HGate` itself.

H = HGate()
I = IGate()
S = SGate()
Sdg = SdgGate()
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
