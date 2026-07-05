"""Dimension-generic gates (generalized Pauli group + controlled add)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class Shift(Operation):
    """Generalized-Pauli cyclic shift: ``|k> -> |(k + power) mod d>``.

    Dimension-free: applies to a subsystem of any dimension; its matrix is
    built from the target dimension at backend lowering. Reduces to X at
    ``dim=2, power=1``.

    Attributes:
        power: Shift amount (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "Shift"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class Clock(Operation):
    """Generalized-Pauli phase: ``|k> -> omega^(k*power) |k>``, omega=e^{2πi/d}.

    Reduces to Z at ``dim=2, power=1``.

    Attributes:
        power: Phase power (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "Clock"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SumGate(Operation):
    """Generalized controlled add: ``|i, j> -> |i, (i + j) mod d>``.

    ``targets = (control, target)``; operand 0 is the control. The default
    implementation requires equal target dimensions.

    The class itself is not part of the `qs.ops` public surface (not in
    `__all__`) but stays attribute-accessible for `isinstance` checks; `Sum`
    (the singleton) is the one users add to a program.
    """

    name: ClassVar[str] = "Sum"
    _num_subsystems: ClassVar[int] = 2


# `Sum` takes no parameters, so - like the fixed gates - it is exported only
# as a singleton value.
Sum = SumGate()
