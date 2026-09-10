"""OpenQASM/frontend Program lowering into immutable logical IR."""

from __future__ import annotations

from ...logical_program import LogicalProgram
from ...operations import Measurement
from ...program import Program, _AppliedOperation
from ...qasm import from_qasm
from ...registers import RegisterRef, RegisterView, _view_members
from ..core import CompileContext
from ..dialects.logical_gate import (
    LogicalGate,
    LogicalIR,
    LogicalMeasure,
)
from ..dialects.qasm import QasmSource
from ..errors import UnsupportedFeatureError


def snapshot_program(program: Program) -> LogicalIR:
    """Freeze a legacy frontend Program into target-independent logical IR."""
    if type(program) is not Program:
        raise TypeError("snapshot_program expects an exact fatqat.Program")
    return _snapshot_program(program)


def _freeze_logical_program(program: LogicalProgram) -> LogicalIR:
    """Freeze an editable compiler frontend into immutable logical IR."""
    if type(program) is not LogicalProgram:
        raise TypeError("freeze pass expects an exact LogicalProgram")
    return _snapshot_program(program)


def _snapshot_program(program: Program) -> LogicalIR:
    qubits = _declared_refs(program.quantum_registers)
    clbits = _declared_refs(program.classical_registers)

    instructions: list[LogicalGate | LogicalMeasure] = []
    for frontend_instruction in program._instructions:
        if type(frontend_instruction) is _AppliedOperation:
            if frontend_instruction.condition is not None:
                raise UnsupportedFeatureError(
                    "classical condition is not supported by static logical IR v0.1"
                )
            for operands in _expanded_operands(frontend_instruction.targets):
                instructions.append(
                    LogicalGate(
                        operation_id=f"logical.{len(instructions)}",
                        operation=frontend_instruction.operation,
                        operands=operands,
                    )
                )
            continue

        if type(frontend_instruction) is not Measurement:
            raise UnsupportedFeatureError(
                f"unsupported frontend instruction: {type(frontend_instruction).__name__}"
            )
        for target, output in zip(
            frontend_instruction.targets, frontend_instruction.outputs
        ):
            instructions.append(
                LogicalMeasure(
                    operation_id=f"logical.{len(instructions)}",
                    qubit=target,
                    clbit=output,
                )
            )

    return LogicalIR(qubits, clbits, tuple(instructions))


def _expanded_operands(targets) -> tuple[tuple[RegisterRef, ...], ...]:
    """Expand Program register views into scalar logical gate occurrences."""
    values = tuple(targets)
    if not any(isinstance(target, RegisterView) for target in values):
        return (values,)
    groups = tuple(_view_members(target) for target in values)
    return tuple(zip(*groups, strict=True))


def _declared_refs(registers) -> tuple[RegisterRef, ...]:
    return tuple(
        register[index] for register in registers for index in range(register.size)
    )


class ParseQasmPass:
    name = "parse-qasm"
    source_type = QasmSource
    target_type = LogicalIR

    def run(self, source: QasmSource, context: CompileContext) -> LogicalIR:
        del context
        return snapshot_program(from_qasm(source.text))


parse_qasm = ParseQasmPass()


class FreezeLogicalPass:
    name = "freeze-logical"
    source_type = LogicalProgram
    target_type = LogicalIR

    def run(self, source: LogicalProgram, context: CompileContext) -> LogicalIR:
        del context
        return _freeze_logical_program(source)


freeze_logical = FreezeLogicalPass()
