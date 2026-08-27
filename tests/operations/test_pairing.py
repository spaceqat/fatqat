"""Tests the Pair / Unpair operations: arity, naming, singleton, program use."""

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.program import AppliedOperation, Program


@pytest.mark.parametrize(
    "singleton, cls, name",
    [(ops.Pair, ops.PairGate, "Pair"), (ops.Unpair, ops.UnpairGate, "Unpair")],
)
def test_singleton_shape(singleton, cls, name):
    assert singleton.name == name
    assert singleton.num_subsystems == 2
    assert isinstance(singleton, cls)
    assert singleton == cls()  # frozen value equality


@pytest.mark.parametrize("op", [ops.Pair, ops.Unpair])
def test_added_to_program_with_two_targets(op):
    p = Program(2)
    p.add(op, (0, 1))
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert step.operation is op
    assert step.targets == (p.quantum_registers[0][0], p.quantum_registers[0][1])


@pytest.mark.parametrize("op", [ops.Pair, ops.Unpair])
def test_wrong_arity_rejected(op):
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(op, 0)  # needs exactly two targets


@pytest.mark.parametrize("op", [ops.Pair, ops.Unpair])
def test_self_pair_rejected_as_duplicate_target(op):
    # Program.add already rejects a repeated target, which is exactly the
    # "no self-loop" rule at the program level.
    p = Program(2)
    with pytest.raises(ValueError, match="more than once"):
        p.add(op, (0, 0))


@pytest.mark.parametrize("op", [ops.Pair, ops.Unpair])
def test_exported_on_operations_namespace(op):
    assert op.name in fq.operations.__all__
    assert getattr(fq.operations, op.name) is op


def test_old_operations_are_gone():
    for removed in ("Rearrange", "LoadAtoms", "Refill"):
        assert removed not in fq.operations.__all__
        assert not hasattr(fq.operations, removed)
