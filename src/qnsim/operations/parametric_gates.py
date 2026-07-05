"""Parametric unitary gates: single-qubit rotations/phase and a controlled phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation

# ---------------------------------------------------------------------------
# Parametric single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RX(Operation):
    """Rotation around the X axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RX"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    """Rotation around the Y axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RY"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    """Rotation around the Z axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RZ"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class Phase(Operation):
    """General single-qubit phase gate: diag(1, e^{i theta}).

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "Phase"
    _num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Parametric controlled / multi-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPhase(Operation):
    """Controlled phase gate: applies diag(1, e^{i theta}) to the target when
    the control is |1>.

    ``targets = (control, target)``; operand 0 is the control.

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "CPhase"
    _num_subsystems: ClassVar[int] = 2
