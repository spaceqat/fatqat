"""Parameterized single-qubit rotations, phase gates, and controlled phase.

Examples:
    ``RX(pi)`` on ``|0>`` matches ``X`` up to the global phase ``-i``:

    >>> import math
    >>> import numpy as np
    >>> import fatqat as fq
    >>> import fatqat.operations as ops
    >>> program = fq.Program(1)
    >>> program.add(ops.RX(math.pi), 0)
    >>> result = fq.simulator.Simulator("SV").run(
    ...     program,
    ...     shots=1,
    ...     result_config={"counts": False, "final_state": True},
    ... ).result()
    >>> np.testing.assert_allclose(
    ...     result.get_statevector(), np.array([0.0, -1.0j]), atol=1e-15
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..parameters import Parameter
from .base import Operation

# ---------------------------------------------------------------------------
# Parametric single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RX(Operation):
    """Rotate a qubit about the X axis by ``theta`` radians.

    In ``|0>, |1>`` basis order, the matrix is
    ``[[c, -i*s], [-i*s, c]]``, where ``c = cos(theta/2)`` and
    ``s = sin(theta/2)``. A `fatqat.RegisterView` applies the same rotation to
    each selected member.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RX"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class RY(Operation):
    """Rotate a qubit about the Y axis by ``theta`` radians.

    In ``|0>, |1>`` basis order, the matrix is ``[[c, -s], [s, c]]``, where
    ``c = cos(theta/2)`` and ``s = sin(theta/2)``. A `fatqat.RegisterView`
    applies the same rotation to each selected member.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RY"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class RZ(Operation):
    """Rotate a qubit about the Z axis by ``theta`` radians.

    The matrix is ``diag(exp(-i*theta/2), exp(i*theta/2))``. It differs from
    `Phase` with the same ``theta`` only by the global phase
    ``exp(-i*theta/2)``. A `fatqat.RegisterView` applies the same rotation to
    each selected member.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RZ"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class Phase(Operation):
    """Multiply a qubit's ``|1>`` amplitude by ``exp(i*theta)``.

    The matrix is ``diag(1, exp(i*theta))``. It differs from `RZ` with the
    same ``theta`` only by a global phase.

    Args:
        theta: Numeric phase angle in radians, or a `fatqat.Parameter` to bind
            before execution.
    """

    theta: float | Parameter
    name: ClassVar[str] = "Phase"
    num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Parametric controlled / multi-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class U(Operation):
    """Apply a general single-qubit U gate.

    In ``|0>, |1>`` basis order, the matrix is
    ``[[c, -exp(i*lam)*s], [exp(i*phi)*s, exp(i*(phi+lam))*c]]``, where
    ``c = cos(theta/2)`` and ``s = sin(theta/2)``. Each angle may be numeric
    or a `fatqat.Parameter` bound before execution.

    Args:
        theta: Polar rotation angle in radians.
        phi: Phase angle in radians.
        lam: Second phase angle in radians. The parameter order is
            ``(theta, phi, lam)``, following Qiskit's ``UGate`` convention.
    """

    theta: float | Parameter
    phi: float | Parameter
    lam: float | Parameter
    name: ClassVar[str] = "U"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U1(Operation):
    """Apply the single-qubit U1 phase gate.

    ``U1(lam)`` has matrix ``diag(1, exp(i*lam))`` and is equivalent to
    `Phase` with ``theta=lam``.

    Args:
        lam: Numeric phase angle in radians, or a `fatqat.Parameter` to bind
            before execution.
    """

    lam: float | Parameter
    name: ClassVar[str] = "U1"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U2(Operation):
    """Apply the single-qubit U2 gate.

    ``U2(phi, lam)`` is equivalent to `U` with ``theta=pi/2`` and the same
    ``phi`` and ``lam``. Both angles may be numeric or `fatqat.Parameter`
    values bound before execution.

    Args:
        phi: Phase angle in radians.
        lam: Second phase angle in radians.
    """

    phi: float | Parameter
    lam: float | Parameter
    name: ClassVar[str] = "U2"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U3(Operation):
    """Apply the single-qubit U3 gate.

    ``U3(theta, phi, lam)`` is matrix-identical to `U` with the same arguments
    and is retained for Qiskit conversion compatibility. Each angle may be
    numeric or a `fatqat.Parameter` bound before execution.

    Args:
        theta: Polar rotation angle in radians.
        phi: Phase angle in radians.
        lam: Second phase angle in radians.
    """

    theta: float | Parameter
    phi: float | Parameter
    lam: float | Parameter
    name: ClassVar[str] = "U3"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class CPhase(Operation):
    """Multiply ``|11>`` by ``exp(i*theta)``.

    Targets are ``(control, target)`` and the matrix in that local basis order
    is ``diag(1, 1, 1, exp(i*theta))``.

    Args:
        theta: Numeric phase angle in radians, or a `fatqat.Parameter` to bind
            before execution.
    """

    theta: float | Parameter
    name: ClassVar[str] = "CPhase"
    num_subsystems: ClassVar[int] = 2
