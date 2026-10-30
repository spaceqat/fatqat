"""Passive linear-optical gates on bosonic modes.

A mode is one slot of a `fatqat.ModeRegister`; its basis level ``n`` is the
photon number. `fatqat.simulator.LOSimulator` truncates every mode to
``cutoff + 1`` levels for each run. The gates also run on fixed-dimension
`fatqat.QuantumRegister` slots, where level ``d - 1`` is the truncation.

Examples:
    Interfere one photon on a balanced beam splitter:

    >>> import math
    >>> import fatqat as fq
    >>> import fatqat.operations as ops
    >>> modes = fq.ModeRegister(2)
    >>> program = fq.Program([modes])
    >>> program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..parameters import Parameter
from ..registers import RegisterRef
from .base import Operation
from .parametric_gates import _validate_angles


@dataclass(frozen=True)
class PhaseShifter(Operation):
    """Apply a photon-number-dependent phase to one mode.

    The ``n``-photon level gains phase ``exp(i*n*phi)``. On a dimension-2
    target this is `Phase` with ``theta=phi``.

    Args:
        phi: Numeric phase in radians, or a `fatqat.Parameter` to bind before
            execution.
    """

    phi: float | Parameter
    name: ClassVar[str] = "PhaseShifter"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_angles(self, phi=self.phi)


@dataclass(frozen=True)
class BeamSplitter(Operation):
    """Mix two modes on a lossless beam splitter.

    Targets are ``(mode_0, mode_1)``. With annihilation operators ``a_0`` and
    ``a_1``, the gate is
    ``exp(theta * (exp(i*phi) * a_1^dag a_0 - exp(-i*phi) * a_0^dag a_1))``.
    It conserves the total photon number. On one photon in the ordered basis
    ``(|1, 0>, |0, 1>)`` it acts as
    ``[[cos(theta), -exp(-i*phi)*sin(theta)], [exp(i*phi)*sin(theta), cos(theta)]]``.

    The reflectivity is ``sin(theta)**2``, so ``theta=pi/4`` is a balanced
    (50:50) splitter and ``theta=asin(sqrt(R))`` gives reflectivity ``R``. For
    a dual-rail qubit with ``|0> = |1, 0>`` and ``|1> = |0, 1>``,
    ``BeamSplitter(a / 2, -pi / 2)`` acts as ``RX(a)`` and
    ``BeamSplitter(a / 2)`` acts as ``RY(a)``.

    A finite truncation keeps the gate unitary and photon-number conserving.
    It is exact when the two modes hold at most as many photons in total as
    one truncated mode can, that is ``cutoff`` photons on
    `fatqat.simulator.LOSimulator` or ``d - 1`` on dimension-``d`` targets.

    Args:
        theta: Numeric mixing angle in radians, or a `fatqat.Parameter` to bind
            before execution.
        phi: Numeric phase in radians (default ``0.0``), or a
            `fatqat.Parameter` to bind before execution.

    Raises:
        ValueError: From `fatqat.Program.add` if the two targets have
            different fixed dimensions.
    """

    theta: float | Parameter
    phi: float | Parameter = 0.0
    name: ClassVar[str] = "BeamSplitter"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_angles(self, theta=self.theta, phi=self.phi)

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Require both resolved targets to share one local dimension.

        Args:
            targets: Two resolved scalar `fatqat.RegisterRef` values.

        Raises:
            ValueError: If the target dimensions differ.
        """
        dims = tuple(target.register.dim for target in targets)
        if dims[0] != dims[1]:
            raise ValueError(
                f"BeamSplitter requires equal target dimensions, got {dims}"
            )
