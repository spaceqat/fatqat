"""Tests the Put operation: variable arity, naming, singleton, program use."""

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.program import _AppliedOperation, Program


def test_singleton_shape():
    assert ops.Put.name == "Put"
    assert ops.Put.num_subsystems is None  # variable arity, >= 1 target
    assert isinstance(ops.Put, ops.Operation)
    assert not isinstance(ops.Put, type)
    assert ops.Put.accepts_views


def test_put_added_with_multiple_targets():
    p = Program(3)
    p.add(ops.Put, (0, 1, 2))
    (step,) = p._instructions
    assert isinstance(step, _AppliedOperation)
    assert step.operation is ops.Put
    assert len(step.targets) == 3


def test_put_added_with_single_target():
    p = Program(2)
    p.add(ops.Put, 0)
    (step,) = p._instructions
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


def test_put_exported_on_operations_namespace():
    assert "Put" in fq.operations.__all__
    assert fq.operations.Put is ops.Put
