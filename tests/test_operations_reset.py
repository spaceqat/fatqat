import pytest

import qnsim as qs
from qnsim import operations as ops
from qnsim.program import AppliedOperation, Program


def test_reset_is_exposed_and_constructible():
    r = qs.ops.Reset()
    assert isinstance(r, ops.Operation)
    assert r.name == "Reset"
    assert r.num_qubits is None


def test_reset_added_to_program_as_applied_operation():
    p = Program(1)
    p.add(qs.ops.Reset(), 0)
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert isinstance(step.operation, ops.Reset)
    assert len(step.targets) == 1


def test_reset_accepts_one_or_many_targets():
    p = qs.Program(3)
    p.add(qs.ops.Reset(), 0)
    p.add(qs.ops.Reset(), (0, 1, 2))

    assert p.operations[0].targets == (p.qreg[0][0],)
    assert p.operations[1].targets == (
        p.qreg[0][0],
        p.qreg[0][1],
        p.qreg[0][2],
    )


def test_reset_rejects_empty_targets():
    p = qs.Program(1)
    with pytest.raises(ValueError, match="at least one target"):
        p.add(qs.ops.Reset(), ())
