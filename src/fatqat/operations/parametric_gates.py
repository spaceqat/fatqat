"""Parametric unitary gates: single-qubit rotations/phase and a controlled phase.

Examples:
    ``RX(pi)`` on ``|0>`` matches ``X`` up to the global phase ``-i``:

    >>> import math
    >>> import fatqat as fq
    >>> import fatqat.operations as ops
    >>> program = fq.Program(1)
    >>> program.add(ops.RX(math.pi), 0)
    >>> result = fq.simulator.Simulator("SV").run(
    ...     program,
    ...     shots=1,
    ...     result_config={"counts": False, "final_state": True},
    ... ).result()
    >>> result.get_statevector()
    array([0.-0.j, 0.-1.j])
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
    ``s = sin(theta/2)``. A ``RegisterView`` applies the same rotation to each
    selected member.

    Args:
        theta: Numeric angle in radians, or a ``fatqat.Parameter`` to bind
            before execution. The value is stored unchanged.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RX"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class RY(Operation):
    """Rotate a qubit about the Y axis by ``theta`` radians.

    In ``|0>, |1>`` basis order, the matrix is ``[[c, -s], [s, c]]``, where
    ``c = cos(theta/2)`` and ``s = sin(theta/2)``. A ``RegisterView`` applies
    the same rotation to each selected member.

    Args:
        theta: Numeric angle in radians, or a ``fatqat.Parameter`` to bind
            before execution. The value is stored unchanged.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RY"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class RZ(Operation):
    """Rotate a qubit about the Z axis by ``theta`` radians.

    The matrix is ``diag(exp(-i*theta/2), exp(i*theta/2))``. It differs from
    ``Phase(theta)`` only by the global phase ``exp(-i*theta/2)``. A
    ``RegisterView`` applies the same rotation to each selected member.

    Args:
        theta: Numeric angle in radians, or a ``fatqat.Parameter`` to bind
            before execution. The value is stored unchanged.
    """

    theta: float | Parameter
    name: ClassVar[str] = "RZ"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class Phase(Operation):
    """Multiply a qubit's ``|1>`` amplitude by ``exp(i*theta)``.

    The matrix is ``diag(1, exp(i*theta))``. It differs from ``RZ(theta)``
    only by a global phase.

    Args:
        theta: Numeric phase angle in radians, or a ``fatqat.Parameter`` to
            bind before execution. The value is stored unchanged.
    """

    theta: float | Parameter
    name: ClassVar[str] = "Phase"
    num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Parametric controlled / multi-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class U(Operation):
    """Apply the Qiskit-compatible general single-qubit U gate.

    In ``|0>, |1>`` basis order, the matrix is
    ``[[c, -exp(i*lam)*s], [exp(i*phi)*s, exp(i*(phi+lam))*c]]``, where
    ``c = cos(theta/2)`` and ``s = sin(theta/2)``.

    Args:
        theta: Polar rotation angle in radians, or a ``fatqat.Parameter`` to
            bind before execution.
        phi: Phase angle in radians, or a ``fatqat.Parameter`` to bind before
            execution.
        lam: Second phase angle in radians, or a ``fatqat.Parameter`` to bind
            before execution. The parameter order is exactly
            ``(theta, phi, lam)``, matching Qiskit's ``UGate``.
    """

    theta: float | Parameter
    phi: float | Parameter
    lam: float | Parameter
    name: ClassVar[str] = "U"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U1(Operation):
    """Apply the legacy Qiskit U1 phase gate.

    ``U1(lam)`` has matrix ``diag(1, exp(i*lam))`` and is equivalent to
    ``Phase(lam)``.

    Args:
        lam: Phase angle in radians, or a ``fatqat.Parameter`` to bind before
            execution.
    """

    lam: float | Parameter
    name: ClassVar[str] = "U1"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U2(Operation):
    """Apply the legacy Qiskit U2 gate.

    ``U2(phi, lam)`` is equivalent to ``U(pi/2, phi, lam)``.

    Args:
        phi: Phase angle in radians, or a ``fatqat.Parameter`` to bind before
            execution.
        lam: Second phase angle in radians, or a ``fatqat.Parameter`` to bind
            before execution.
    """

    phi: float | Parameter
    lam: float | Parameter
    name: ClassVar[str] = "U2"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class U3(Operation):
    """Apply the legacy Qiskit U3 gate.

    ``U3(theta, phi, lam)`` is numerically identical to
    ``U(theta, phi, lam)`` and retains the legacy operation name for
    conversion compatibility.

    Args:
        theta: Polar rotation angle in radians, or a ``fatqat.Parameter`` to
            bind before execution.
        phi: Phase angle in radians, or a ``fatqat.Parameter`` to bind before
            execution.
        lam: Second phase angle in radians, or a ``fatqat.Parameter`` to bind
            before execution.
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
        theta: Numeric phase angle in radians, or a ``fatqat.Parameter`` to
            bind before execution. The value is stored unchanged.
    """

    theta: float | Parameter
    name: ClassVar[str] = "CPhase"
    num_subsystems: ClassVar[int] = 2
