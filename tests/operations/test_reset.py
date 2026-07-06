import pytest

import fatqat as fc
from fatqat import operations as ops
from fatqat.program import AppliedOperation, Program


def test_reset_is_a_non_callable_singleton():
    assert isinstance(fc.ops.Reset, ops.ResetGate)
    assert isinstance(fc.ops.Reset, ops.Operation)
    assert fc.ops.Reset.name == "Reset"
    assert fc.ops.Reset.num_subsystems is None
    with pytest.raises(TypeError):
        fc.ops.Reset()


def test_reset_added_to_program_as_applied_operation():
    p = Program(1)
    p.add(fc.ops.Reset, 0)
    (step,) = p.operations
    assert isinstance(step, AppliedOperation)
    assert isinstance(step.operation, ops.ResetGate)
    assert len(step.targets) == 1


def test_reset_accepts_one_or_many_targets():
    p = fc.Program(3)
    p.add(fc.ops.Reset, 0)
    p.add(fc.ops.Reset, (0, 1, 2))

    assert p.operations[0].targets == (p.qreg[0][0],)
    assert p.operations[1].targets == (
        p.qreg[0][0],
        p.qreg[0][1],
        p.qreg[0][2],
    )


def test_reset_rejects_empty_targets():
    p = fc.Program(1)
    with pytest.raises(ValueError, match="at least one target"):
        p.add(fc.ops.Reset, ())
