"""OpenQASM/frontend Program lowering into immutable logical IR."""

from __future__ import annotations

from ...operations import Measurement
from ...program import Program, _AppliedOperation
from ...qasm import from_qasm
from ...registers import RegisterRef, RegisterView
from ..core import CompileContext
from ..dialects.logical_gate import (
    LogicalGate,
    LogicalMeasure,
    LogicalProgram,
)
from ..dialects.qasm import QasmSource
from ..errors import UnsupportedFeatureError


def snapshot_program(program: Program) -> LogicalProgram:
    """Freeze a frontend Program into deterministic, target-independent logical IR."""

    if type(program) is not Program:
        raise TypeError("snapshot_program expects an exact fatqat.Program")

    qubits = _declared_refs(program.quantum_registers)
    clbits = _declared_refs(program.classical_registers)

    instructions: list[LogicalGate | LogicalMeasure] = []
    for frontend_instruction in program._instructions:
        if type(frontend_instruction) is _AppliedOperation:
            if frontend_instruction.condition is not None:
                raise UnsupportedFeatureError(
                    "classical condition is not supported by static logical IR v0.1"
                )
            if any(
                isinstance(item, RegisterView) for item in frontend_instruction.targets
            ):
                raise UnsupportedFeatureError(
                    "RegisterView operations must be expanded before compiler snapshot"
                )
            operands = tuple(frontend_instruction.targets)
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

    return LogicalProgram(qubits, clbits, tuple(instructions))


def _declared_refs(registers) -> tuple[RegisterRef, ...]:
    return tuple(
        register[index] for register in registers for index in range(register.size)
    )


class ParseQasmPass:
    name = "parse-qasm"
    source_type = QasmSource
    target_type = LogicalProgram

    def run(self, source: QasmSource, context: CompileContext) -> LogicalProgram:
        del context
        return snapshot_program(from_qasm(source.text))


parse_qasm = ParseQasmPass()
