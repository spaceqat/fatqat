"""Parametric unitary gates: single-qubit rotations/phase and a controlled phase.

Examples:
    ``RX(pi)`` on ``|0>`` matches ``X`` up to the global phase ``-i``:

    >>> import math
    >>> import fatqat as fq
    >>> program = fq.Program(1)
    >>> program.add(fq.ops.RX(math.pi), 0)
    >>> result = fq.backends.StateVectorBackend().run(
    ...     program,
    ...     shots=1,
    ...     result_config={"counts": False, "statevector": True},
    ... ).result()
    >>> result.get_statevector()
    array([0.-0.j, 0.-1.j])
"""

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

    .. math::

        R_X(\\theta) = \\begin{pmatrix}
        \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
        -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RX"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    """Rotation around the Y axis.

    .. math::

        R_Y(\\theta) = \\begin{pmatrix}
        \\cos\\frac{\\theta}{2} & -\\sin\\frac{\\theta}{2} \\\\
        \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RY"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    """Rotation around the Z axis.

    .. math::

        R_Z(\\theta) = \\begin{pmatrix}
        e^{-i\\theta/2} & 0 \\\\ 0 & e^{i\\theta/2}
        \\end{pmatrix}

    Note this differs from :class:`Phase` by the global phase
    :math:`e^{-i\\theta/2}`.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RZ"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class Phase(Operation):
    """General single-qubit phase gate: ``diag(1, e^{i theta})``.

    .. math::

        \\mathrm{Phase}(\\theta) = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\theta} \\end{pmatrix}

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
    """Controlled phase gate: applies ``diag(1, e^{i theta})`` to the target
    when the control is ``|1>``.

    ``targets = (control, target)``; operand 0 is the control. Basis order
    :math:`|00\\rangle, |01\\rangle, |10\\rangle, |11\\rangle`.

    .. math::

        \\mathrm{CPhase}(\\theta) = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&1&0\\\\ 0&0&0&e^{i\\theta}
        \\end{pmatrix}

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "CPhase"
    _num_subsystems: ClassVar[int] = 2
