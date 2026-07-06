"""Tests AppliedOperation and Measurement value object validation."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fatqcat.registers import QuantumRegister, ClassicalRegister
from fatqcat import operations as ops
from fatqcat.operations import Measurement
from fatqcat.program import AppliedOperation


def test_applied_operation_accepts_correct_arity():
    qr = QuantumRegister(2)
    ao = AppliedOperation(operation=ops.CX, targets=(qr[0], qr[1]))
    assert ao.operation is ops.CX
    assert ao.targets == (qr[0], qr[1])
    assert ao.condition is None


def test_applied_operation_wrong_arity_raises():
    qr = QuantumRegister(2)
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.X, targets=(qr[0], qr[1]))  # X is 1-qubit
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.CX, targets=(qr[0],))  # CX is 2-qubit


def test_applied_operation_rejects_duplicate_targets():
    qr = QuantumRegister(1)
    with pytest.raises(ValueError, match="appears more than once"):
        AppliedOperation(operation=ops.CX, targets=(qr[0], qr[0]))


def test_measurement_fields():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    m = Measurement(qreg=(qr[0],), clreg=(cr[0],))
    assert m.qreg == (qr[0],)
    assert m.clreg == (cr[0],)


def test_validate_targets_default_is_noop():
    qr = QuantumRegister(2)
    ao = AppliedOperation(operation=ops.CX, targets=(qr[0], qr[1]))
    assert ao.operation is ops.CX  # constructing did not raise


def test_validate_targets_hook_is_called_with_resolved_targets():
    @dataclass(frozen=True)
    class _Probe(ops.Operation):
        name: ClassVar[str] = "Probe"
        _num_subsystems: ClassVar[int] = 1

        def validate_targets(self, targets):
            raise ValueError(f"probe saw dim {targets[0].register.dim}")

    qr = QuantumRegister(1, dim=3)
    with pytest.raises(ValueError, match="probe saw dim 3"):
        AppliedOperation(operation=_Probe(), targets=(qr[0],))
