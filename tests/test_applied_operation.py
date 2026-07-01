"""Tests AppliedOperation and Measurement value object validation."""

import pytest

from qnsim.registers import QuantumRegister, ClassicalRegister
from qnsim import operations as ops
from qnsim.program import AppliedOperation, Measurement


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


def test_applied_operation_targets_must_be_quantum():
    cr = ClassicalRegister(1)
    with pytest.raises(TypeError):
        AppliedOperation(operation=ops.X, targets=(cr[0],))


def test_applied_operation_targets_must_be_tuple():
    qr = QuantumRegister(1)
    with pytest.raises(TypeError):
        AppliedOperation(operation=ops.X, targets=[qr[0]])  # list, not tuple


def test_measurement_fields():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    m = Measurement(qreg=qr[0], clreg=cr[0])
    assert m.qreg == qr[0]
    assert m.clreg == cr[0]
