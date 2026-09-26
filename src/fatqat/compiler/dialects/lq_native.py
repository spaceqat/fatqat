"""Physical CZ-native program for LogicalQubit Cloud devices."""

from dataclasses import dataclass, field
from typing import ClassVar

from ...operations.fixed_gates import CZGate, HGate
from ...operations.parametric_gates import RZ
from ...registers import ClassicalRegister
from ..algorithms.sabre import LayoutSnapshot
from .sc_native import NativeInstruction, _verify_native_program


@dataclass(frozen=True, slots=True)
class LQNativeProgram:
    """H/RZ/CZ instructions on fixed physical sites, with logical layouts.

    Measurement destinations retain original classical register identities.
    ``classical_registers`` preserves source declaration order and unused slots.
    """

    IR_ID: ClassVar[str] = "sc.lqcloud.native.v1"

    operations: tuple[NativeInstruction, ...]
    initial_layout: LayoutSnapshot
    final_layout: LayoutSnapshot
    classical_registers: tuple[ClassicalRegister, ...] | None = field(
        default=None, kw_only=True
    )


def verify_lq_native_program(program: object) -> None:
    """Verify a LogicalQubit native program's gates, references and provenance."""
    _verify_native_program(program, LQNativeProgram, (HGate, RZ, CZGate), "LQ")
