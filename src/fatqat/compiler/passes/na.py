"""Logical gate normalization into the closed neutral-atom gate IR."""

from __future__ import annotations

import math

from ... import operations as ops
from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import Phase, RX, RY, RZ
from ...registers import RegisterRef
from ..core import CompileContext
from ..dialects.logical_gate import LogicalGate, LogicalMeasure, LogicalProgram
from ..dialects.na_gate import NAGate, NAMeasure, NAProgram
from ..errors import UnsupportedFeatureError


class _NABuilder:
    """Append-only builder for deterministic neutral-atom instructions."""

    def __init__(
        self, atoms: tuple[RegisterRef, ...], clbits: tuple[RegisterRef, ...]
    ) -> None:
        self.atoms = atoms
        self.clbits = clbits
        self._instructions: list[NAGate | NAMeasure] = []

    def add_gate(
        self,
        operation: RX | RY | RZ | CZGate,
        atoms: tuple[RegisterRef, ...],
        origin_id: str,
    ) -> None:
        self._instructions.append(
            NAGate(
                operation_id=f"na.{len(self._instructions)}",
                origin_ids=(origin_id,),
                operation=operation,
                atoms=atoms,
            )
        )

    def add_measure(
        self, atom: RegisterRef, clbit: RegisterRef, origin_id: str
    ) -> None:
        self._instructions.append(
            NAMeasure(
                operation_id=f"na.{len(self._instructions)}",
                origin_ids=(origin_id,),
                atom=atom,
                clbit=clbit,
            )
        )

    def freeze(self) -> NAProgram:
        return NAProgram(self.atoms, self.clbits, tuple(self._instructions))


def normalize_na_program(source: LogicalProgram) -> NAProgram:
    """Lower numeric, static logical gates into the neutral-atom gate set."""

    builder = _NABuilder(source.qubits, source.clbits)
    measurement_started = False

    for instruction in source.instructions:
        if type(instruction) is LogicalMeasure:
            measurement_started = True
            builder.add_measure(
                instruction.qubit, instruction.clbit, instruction.operation_id
            )
            continue
        if type(instruction) is not LogicalGate:
            raise UnsupportedFeatureError(
                f"unsupported logical instruction: {type(instruction).__name__}"
            )
        if measurement_started:
            raise UnsupportedFeatureError(
                "nonterminal logical measurement is not supported by NA normalization"
            )
        _lower_gate(builder, instruction)

    return builder.freeze()


def _lower_gate(builder: _NABuilder, instruction: LogicalGate) -> None:
    operation = instruction.operation
    atoms = instruction.operands
    origin_id = instruction.operation_id

    if type(operation) in (RX, RY, RZ, CZGate):
        builder.add_gate(operation, atoms, origin_id)
        return
    if type(operation) is type(ops.Reset):
        raise UnsupportedFeatureError("Reset is not supported by NA normalization")
    if operation is ops.I:
        return
    if operation is ops.H:
        _add_h(builder, atoms[0], origin_id)
        return
    if operation is ops.X:
        builder.add_gate(ops.RX(math.pi), atoms, origin_id)
        return
    if operation is ops.Y:
        builder.add_gate(ops.RY(math.pi), atoms, origin_id)
        return
    if operation is ops.Z:
        builder.add_gate(ops.RZ(math.pi), atoms, origin_id)
        return
    if operation is ops.S:
        builder.add_gate(ops.RZ(math.pi / 2), atoms, origin_id)
        return
    if operation is ops.Sdg:
        builder.add_gate(ops.RZ(-math.pi / 2), atoms, origin_id)
        return
    if operation is ops.T:
        builder.add_gate(ops.RZ(math.pi / 4), atoms, origin_id)
        return
    if operation is ops.Tdg:
        builder.add_gate(ops.RZ(-math.pi / 4), atoms, origin_id)
        return
    if type(operation) is Phase:
        builder.add_gate(ops.RZ(operation.theta), atoms, origin_id)
        return
    if operation is ops.CX:
        _add_cx(builder, atoms[0], atoms[1], origin_id)
        return
    if type(operation) is SwapGate:
        first, second = atoms
        _add_cx(builder, first, second, origin_id)
        _add_cx(builder, second, first, origin_id)
        _add_cx(builder, first, second, origin_id)
        return
    raise UnsupportedFeatureError(
        f"operation {operation.name} is not supported by NA normalization"
    )


def _add_h(builder: _NABuilder, atom: RegisterRef, origin_id: str) -> None:
    builder.add_gate(ops.RZ(math.pi), (atom,), origin_id)
    builder.add_gate(ops.RY(math.pi / 2), (atom,), origin_id)


def _add_cx(
    builder: _NABuilder, control: RegisterRef, target: RegisterRef, origin_id: str
) -> None:
    _add_h(builder, target, origin_id)
    builder.add_gate(ops.CZ, (control, target), origin_id)
    _add_h(builder, target, origin_id)


class NormalizeNaPass:
    name = "normalize-na"
    source_type = LogicalProgram
    target_type = NAProgram

    def run(self, source: LogicalProgram, context: CompileContext) -> NAProgram:
        del context
        return normalize_na_program(source)


normalize_na = NormalizeNaPass()
