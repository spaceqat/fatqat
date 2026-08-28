"""Tests Program.measure insertion, ordering, and validation."""

import pytest

from fatqat.program import Program
from fatqat.operations import Measurement
import fatqat.operations as ops
from fatqat.registers import GridRegister, QuantumRegister, ClassicalRegister


def test_measure_appends_measurement():
    p = Program(2, 2)
    result = p.measure(0, 0)
    assert result is None
    assert len(p._instructions) == 1
    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0],)
    assert m.outputs == (p.classical_registers[0][0],)


def test_instructions_preserve_order_and_type_mix():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.measure(0, 0)
    p.measure(1, 1)
    assert len(p._instructions) == 4
    assert p._instructions[0].operation.name == "H"
    assert isinstance(p._instructions[2], Measurement)


def test_measure_rejects_quantum_ref_as_output():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.measure(0, p.quantum_registers[0][1])  # quantum ref as classical slot


def test_measure_int_output_rejects_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.measure(0, 0)


def test_measure_explicit_output_ref_works_with_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    p.measure(1, p.classical_registers[1][0])

    assert p._instructions[0].targets == (p.quantum_registers[0][1],)
    assert p._instructions[0].outputs == (p.classical_registers[1][0],)


def test_measure_rejects_metadata_argument():
    p = Program(1, 1)

    with pytest.raises(TypeError):
        p.measure(0, 0, metadata={"k": 1})


def test_measure_stores_single_as_one_tuples():
    p = Program(2, 2)
    p.measure(0, 1)

    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0],)
    assert m.outputs == (p.classical_registers[0][1],)


def test_measure_accepts_grouped_operands():
    p = Program(3, 3)
    p.measure((0, 2), (1, 0))

    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0], p.quantum_registers[0][2])
    assert m.outputs == (p.classical_registers[0][1], p.classical_registers[0][0])


def test_measure_rejects_mismatched_group_sizes():
    p = Program(3, 2)
    with pytest.raises(ValueError, match="same number"):
        p.measure((0, 1, 2), (0, 1))


def test_measure_rejects_empty_group():
    p = Program(1, 1)
    with pytest.raises(ValueError, match="at least one"):
        p.measure((), ())


def test_measure_all_appends_one_grouped_instruction_in_flat_order():
    p = Program(
        [QuantumRegister(2, name="qa"), QuantumRegister(1, name="qb")],
        [ClassicalRegister(1, name="ca"), ClassicalRegister(2, name="cb")],
    )

    result = p.measure_all()

    assert result is None
    assert len(p._instructions) == 1
    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
        p.quantum_registers[1][0],
    )
    assert m.outputs == (
        p.classical_registers[0][0],
        p.classical_registers[1][0],
        p.classical_registers[1][1],
    )


def test_measure_all_rejects_mismatched_resource_counts():
    p = Program(2, 1)
    with pytest.raises(ValueError, match="same number"):
        p.measure_all()


def test_measure_all_rejects_empty_program():
    p = Program(0, 0)
    with pytest.raises(ValueError, match="at least one"):
        p.measure_all()


def test_measure_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    cb = ClassicalRegister(1, dim=2)
    program = Program([qt], [cb])
    with pytest.raises(ValueError):
        program.measure(qt[0], cb[0])


def test_measure_matching_dims_ok():
    qt = QuantumRegister(1, dim=3)
    ct = ClassicalRegister(1, dim=3)
    program = Program([qt], [ct])
    program.measure(qt[0], ct[0])  # no raise


def test_measure_rejects_register_view_target():
    qubits = GridRegister(2, 2, name="qubits")
    cr = ClassicalRegister(4)
    p = Program([qubits], [cr])
    with pytest.raises(TypeError):
        p.measure(qubits.row(0), 0)


def test_measure_all_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    qb = QuantumRegister(1, dim=2)
    cb = ClassicalRegister(1, dim=2)
    ct = ClassicalRegister(1, dim=3)
    # quantum order (3, 2) vs classical order (2, 3): flat pairing mismatches.
    program = Program([qt, qb], [cb, ct])
    with pytest.raises(ValueError):
        program.measure_all()
