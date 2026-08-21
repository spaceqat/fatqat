"""Tests the Put operation: variable arity, naming, singleton, program use."""

import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.program import AppliedOperation, Program


def test_singleton_shape():
    assert ops.Put.name == "Put"
    assert ops.Put.num_targets is None  # variable arity, >= 1 target
    assert isinstance(ops.Put, ops.PutGate)
    assert ops.Put == ops.PutGate()


def test_put_added_with_multiple_targets():
    p = Program(3)
    p.add(ops.Put, (0, 1, 2))
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert step.operation is ops.Put
    assert len(step.targets) == 3


def test_put_added_with_single_target():
    p = Program(2)
    p.add(ops.Put, 0)
    (step,) = p.operations
    assert step.operation is ops.Put
    assert step.targets == (p.quantum_registers[0][0],)


def test_put_requires_at_least_one_target():
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(ops.Put)  # variable arity still needs >= 1 target


def test_put_rejects_duplicate_targets():
    p = Program(2)
    with pytest.raises(ValueError, match="more than once"):
        p.add(ops.Put, (0, 0))


def test_put_exported_on_fq_ops():
    assert "Put" in fq.ops.__all__
    assert fq.ops.Put is ops.Put
