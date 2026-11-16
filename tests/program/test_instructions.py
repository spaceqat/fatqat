"""Tests for the public Program.instructions interface."""

import fatqat as fq
import fatqat.operations as ops
from fatqat.program import OperationInstruction


def test_instructions():
    program = fq.Program(2, 1)
    target = program.quantum_registers[0][1]
    condition = ((program.classical_registers[0][0], 1),)

    program.add(ops.H, target, condition=condition)
    program.measure(1, 0)

    operation, measurement = program.instructions

    assert isinstance(operation, OperationInstruction)
    assert operation.operation is ops.H
    assert operation.targets == (target,)
    assert operation.condition == condition

    assert isinstance(measurement, ops.Measurement)
    assert measurement.targets[0] == target
    assert measurement.outputs[0] == program.classical_registers[0][0]


def test_instructions_is_snapshot():
    program = fq.Program(1)
    program.add(ops.H, 0)

    snapshot = program.instructions
    program.add(ops.X, 0)

    assert len(snapshot) == 1
    assert len(program.instructions) == 2
