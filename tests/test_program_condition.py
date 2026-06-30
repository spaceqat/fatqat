import pytest

from qnsim.program import Program
from qnsim import operations as ops
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_single_condition_normalized_to_and_list():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(0, 1))
    cond = p.operations[0].condition
    assert cond == ((p.creg[0][0], 1),)


def test_single_condition_with_ref():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(p.creg[0][1], 0))
    assert p.operations[0].condition == ((p.creg[0][1], 0),)


def test_multiple_conditions_are_conjunction():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=((0, 1), (1, 0)))
    cond = p.operations[0].condition
    assert cond == ((p.creg[0][0], 1), (p.creg[0][1], 0))


def test_no_condition_is_none():
    p = Program(2, 2)
    p.add(ops.X, 0)
    assert p.operations[0].condition is None


def test_condition_rejects_quantum_ref_slot():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add(ops.X, 0, condition=(p.qreg[0][1], 1))


def test_condition_int_slot_rejects_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1)],
        clreg=[ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0, condition=(0, 1))


def test_condition_explicit_slot_refs_work_across_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1)],
        clreg=[ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    p.add(ops.X, 0, condition=(p.creg[1][0], 1))

    assert p.operations[0].condition == ((p.creg[1][0], 1),)
