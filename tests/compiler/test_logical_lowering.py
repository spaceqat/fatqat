import pytest

import fatqat as fq
from fatqat.compiler import (
    CompileContext,
    UnsupportedFeatureError,
    ValidationError,
)
from fatqat.compiler.dialects import (
    LogicalGate,
    LogicalMeasure,
    LogicalProgram,
    QasmSource,
    verify_logical_program,
)
from fatqat.compiler.passes import parse_qasm, snapshot_program


def test_program_snapshot_has_deterministic_refs_and_operation_ids():
    qreg = fq.QuantumRegister(2, name="data")
    creg = fq.ClassicalRegister(2, name="out")
    program = fq.Program([qreg], [creg])
    program.add(fq.operations.RX(0.25), qreg[1])
    program.add(fq.operations.CZ, (qreg[1], qreg[0]))
    program.measure((qreg[0], qreg[1]), (creg[0], creg[1]))

    first = snapshot_program(program)
    second = snapshot_program(program)

    assert first == second
    assert first.qubits == (qreg[0], qreg[1])
    assert first.clbits == (creg[0], creg[1])
    assert tuple(item.operation_id for item in first.instructions) == (
        "logical.0",
        "logical.1",
        "logical.2",
        "logical.3",
    )


def test_program_snapshot_preserves_operation_values_and_operand_order():
    program = fq.Program(2)
    rx = fq.operations.RX(0.5)
    program.add(rx, 1)
    program.add(fq.operations.CZ, (1, 0))

    logical = snapshot_program(program)

    assert isinstance(logical.instructions[0], LogicalGate)
    assert logical.instructions[0].operation is rx
    assert logical.instructions[1].operands == (
        program.quantum_registers[0][1],
        program.quantum_registers[0][0],
    )


def test_grouped_measurement_is_expanded_to_scalar_logical_measurements():
    program = fq.Program(2, 2)
    program.measure((0, 1), (0, 1))

    logical = snapshot_program(program)

    assert all(isinstance(item, LogicalMeasure) for item in logical.instructions)
    assert [item.qubit for item in logical.instructions] == [
        program.quantum_registers[0][0],
        program.quantum_registers[0][1],
    ]
    assert [item.clbit for item in logical.instructions] == [
        program.classical_registers[0][0],
        program.classical_registers[0][1],
    ]


def test_program_snapshot_rejects_classical_conditions():
    program = fq.Program(1, 1)
    program.add(fq.operations.X, 0, condition=(0, 1))

    with pytest.raises(UnsupportedFeatureError, match="condition"):
        snapshot_program(program)


def test_parse_qasm_pass_produces_a_valid_logical_program():
    source = QasmSource("""
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        rx(0.25) q[0];
        cz q[0], q[1];
        c = measure q;
        """)

    logical = parse_qasm.run(source, CompileContext())

    assert isinstance(logical, LogicalProgram)
    assert tuple(item.operation_id for item in logical.instructions) == (
        "logical.0",
        "logical.1",
        "logical.2",
        "logical.3",
    )


def test_snapshot_preserves_register_identity_when_names_collide():
    first = fq.QuantumRegister(1, name="q")
    second = fq.QuantumRegister(1, name="q")
    program = fq.Program([first, second])
    program.add(fq.operations.CZ, (first[0], second[0]))
    logical = snapshot_program(program)

    assert logical.qubits == (first[0], second[0])
    assert logical.qubits[0].register is first
    assert logical.qubits[1].register is second


def test_snapshot_preserves_anonymous_registers_and_dimension():
    register = fq.QuantumRegister(1, dim=3)
    logical = snapshot_program(fq.Program([register]))

    assert logical.qubits == (register[0],)
    assert logical.qubits[0].register.dim == 3


def test_logical_validator_checks_operation_arity():
    q0 = fq.QuantumRegister(1, name="q")[0]
    program = LogicalProgram(
        qubits=(q0,),
        clbits=(),
        instructions=(LogicalGate("logical.0", fq.operations.CZ, (q0,)),),
    )

    with pytest.raises(ValidationError, match="expects 2 operand"):
        verify_logical_program(program)
