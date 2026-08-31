"""Tests Program.add condition normalization and validation."""

import pytest

from fatqat.program import Program
import fatqat.operations as ops
from fatqat.registers import QuantumRegister, ClassicalRegister


def test_single_condition_normalized_to_and_list():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(0, 1))
    cond = p._instructions[0].condition
    assert cond == ((p.classical_registers[0][0], 1),)


def test_single_condition_with_ref():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(p.classical_registers[0][1], 0))
    assert p._instructions[0].condition == ((p.classical_registers[0][1], 0),)


def test_multiple_conditions_are_conjunction():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=((0, 1), (1, 0)))
    cond = p._instructions[0].condition
    assert cond == ((p.classical_registers[0][0], 1), (p.classical_registers[0][1], 0))


def test_no_condition_is_none():
    p = Program(2, 2)
    p.add(ops.X, 0)
    assert p._instructions[0].condition is None


def test_condition_rejects_quantum_ref_slot():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add(ops.X, 0, condition=(p.quantum_registers[0][1], 1))


def test_condition_int_slot_rejects_multiple_classical_registers():
    p = Program(
        [QuantumRegister(1)],
        [ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0, condition=(0, 1))


def test_condition_explicit_slot_refs_work_across_multiple_classical_registers():
    p = Program(
        [QuantumRegister(1)],
        [ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    p.add(ops.X, 0, condition=(p.classical_registers[1][0], 1))

    assert p._instructions[0].condition == ((p.classical_registers[1][0], 1),)


def test_empty_condition_raises_valueerror():
    p = Program(2, 2)
    with pytest.raises(ValueError, match="condition is empty"):
        p.add(ops.X, 0, condition=())


def test_condition_literal_out_of_range_raises():
    qt = QuantumRegister(1, dim=3)
    ct = ClassicalRegister(1, dim=3)
    program = Program([qt], [ct])
    with pytest.raises(ValueError):
        program.add(ops.Shift(1), qt[0], condition=(ct[0], 7))  # 7 >= dim 3


def test_condition_literal_in_range_ok():
    qt = QuantumRegister(1, dim=3)
    ct = ClassicalRegister(1, dim=3)
    program = Program([qt], [ct])
    program.add(ops.Shift(1), qt[0], condition=(ct[0], 2))  # 2 < dim 3, ok


def test_dim2_condition_literal_two_still_rejected():
    p = Program(2, 2)
    with pytest.raises(ValueError):
        p.add(ops.X, 0, condition=(0, 2))


def test_condition_literal_accepts_bool_as_int():
    # Deliberate: a condition literal is boolean in spirit for a dim=2 clbit,
    # so True/False are legitimate spellings of 1/0, unlike the strict
    # int-only fields elsewhere (register size, dim, index).
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(0, True))
    assert p._instructions[0].condition == ((p.classical_registers[0][0], 1),)


def test_condition_literal_rejects_non_int():
    p = Program(2, 2)
    with pytest.raises(TypeError, match="condition literal must be int"):
        p.add(ops.X, 0, condition=(0, 1.5))


def test_condition_mapping_rejected_with_typeerror():
    program = Program(1, 1)
    creg = program.classical_registers[0]
    with pytest.raises(TypeError, match="condition must be"):
        program.add(ops.X, 0, condition={creg[0]: 1})


def test_condition_set_rejected_with_typeerror():
    program = Program(1, 1)
    creg = program.classical_registers[0]
    with pytest.raises(TypeError, match="condition must be"):
        program.add(ops.X, 0, condition={(creg[0], 1)})
