import pytest

import qnsim as qs
from qnsim import operations as ops
from qnsim.program import AppliedOperation, Program


def test_reset_is_exposed_and_constructible():
    r = qs.ops.Reset()
    assert isinstance(r, ops.Operation)
    assert r.name == "Reset"
    assert r.num_qubits == 1


def test_reset_added_to_program_as_applied_operation():
    p = Program(1)
    p.add(qs.ops.Reset(), 0)
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert isinstance(step.operation, ops.Reset)
    assert len(step.targets) == 1


def test_reset_arity_is_one():
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(qs.ops.Reset(), (0, 1))
