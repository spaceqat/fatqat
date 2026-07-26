"""Tests the LoadAtoms operation: construction validation and zero-arity shape."""

import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.program import AppliedOperation, Program


def test_load_atom_stores_rows_and_cols():
    op = ops.LoadAtoms(2, 3)
    assert op.rows == 2
    assert op.cols == 3
    assert op.name == "LoadAtoms"
    assert op.num_targets == 0


def test_load_atom_rejects_non_int_rows():
    with pytest.raises(TypeError):
        ops.LoadAtoms(2.0, 3)


def test_load_atom_rejects_non_int_cols():
    with pytest.raises(TypeError):
        ops.LoadAtoms(2, "3")


def test_load_atom_rejects_bool_rows():
    with pytest.raises(TypeError):
        ops.LoadAtoms(True, 3)


def test_load_atom_rejects_bool_cols():
    with pytest.raises(TypeError):
        ops.LoadAtoms(2, False)


def test_load_atom_rejects_zero_rows():
    with pytest.raises(ValueError):
        ops.LoadAtoms(0, 3)


def test_load_atom_rejects_negative_cols():
    with pytest.raises(ValueError):
        ops.LoadAtoms(2, -1)


def test_load_atom_added_to_program_with_no_targets():
    p = Program(4)
    p.add(fq.ops.LoadAtoms(2, 2))
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert isinstance(step.operation, ops.LoadAtoms)
    assert step.targets == ()


def test_load_atom_is_exported_on_fq_ops():
    assert "LoadAtoms" in fq.ops.__all__
    assert fq.ops.LoadAtoms is ops.LoadAtoms
