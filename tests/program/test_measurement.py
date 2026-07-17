"""Tests Program.add_measurement insertion, ordering, and validation."""

import pytest

from fatqat.program import Program
from fatqat.operations import Measurement
from fatqat import operations as ops
from fatqat.registers import QuantumRegister, ClassicalRegister


def test_add_measurement_appends_measurement():
    p = Program(2, 2)
    result = p.add_measurement(0, 0)
    assert result is None
    assert len(p.operations) == 1
    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.qreg[0][0],)
    assert m.outputs == (p.creg[0][0],)


def test_operations_preserve_order_and_type_mix():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    assert len(p.operations) == 4
    assert p.operations[0].operation.name == "H"
    assert isinstance(p.operations[2], Measurement)


def test_add_measurement_rejects_quantum_ref_as_output():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add_measurement(0, p.qreg[0][1])  # quantum ref as classical slot


def test_add_measurement_int_output_rejects_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add_measurement(0, 0)


def test_add_measurement_explicit_output_ref_works_with_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    p.add_measurement(1, p.creg[1][0])

    assert p.operations[0].targets == (p.qreg[0][1],)
    assert p.operations[0].outputs == (p.creg[1][0],)


def test_add_measurement_rejects_metadata_argument():
    p = Program(1, 1)

    with pytest.raises(TypeError):
        p.add_measurement(0, 0, metadata={"k": 1})


def test_add_measurement_stores_single_as_one_tuples():
    p = Program(2, 2)
    p.add_measurement(0, 1)

    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.qreg[0][0],)
    assert m.outputs == (p.creg[0][1],)


def test_add_measurement_accepts_grouped_operands():
    p = Program(3, 3)
    p.add_measurement((0, 2), (1, 0))

    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.qreg[0][0], p.qreg[0][2])
    assert m.outputs == (p.creg[0][1], p.creg[0][0])


def test_add_measurement_rejects_mismatched_group_sizes():
    p = Program(3, 2)
    with pytest.raises(ValueError, match="same number"):
        p.add_measurement((0, 1, 2), (0, 1))


def test_add_measurement_rejects_empty_group():
    p = Program(1, 1)
    with pytest.raises(ValueError, match="at least one"):
        p.add_measurement((), ())


def test_measure_all_appends_one_grouped_instruction_in_flat_order():
    p = Program(
        [QuantumRegister(2, name="qa"), QuantumRegister(1, name="qb")],
        [ClassicalRegister(1, name="ca"), ClassicalRegister(2, name="cb")],
    )

    result = p.measure_all()

    assert result is None
    assert len(p.operations) == 1
    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.qreg[0][0], p.qreg[0][1], p.qreg[1][0])
    assert m.outputs == (p.creg[0][0], p.creg[1][0], p.creg[1][1])


def test_measure_all_rejects_mismatched_resource_counts():
    p = Program(2, 1)
    with pytest.raises(ValueError, match="same number"):
        p.measure_all()


def test_measure_all_rejects_empty_program():
    p = Program(0, 0)
    with pytest.raises(ValueError, match="at least one"):
        p.measure_all()


def test_add_measurement_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    cb = ClassicalRegister(1, dim=2)
    program = Program([qt], [cb])
    with pytest.raises(ValueError):
        program.add_measurement(qt[0], cb[0])


def test_add_measurement_matching_dims_ok():
    qt = QuantumRegister(1, dim=3)
    ct = ClassicalRegister(1, dim=3)
    program = Program([qt], [ct])
    program.add_measurement(qt[0], ct[0])  # no raise


def test_measure_all_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    qb = QuantumRegister(1, dim=2)
    cb = ClassicalRegister(1, dim=2)
    ct = ClassicalRegister(1, dim=3)
    # quantum order (3, 2) vs classical order (2, 3): flat pairing mismatches.
    program = Program([qt, qb], [cb, ct])
    with pytest.raises(ValueError):
        program.measure_all()
