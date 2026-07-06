"""Tests Program.add operation insertion, target resolution, and validation."""

import pytest

from fatqat.program import Program, AppliedOperation
from fatqat import operations as ops
from fatqat.registers import QuantumRegister


def test_add_single_operand_int():
    p = Program(2)
    result = p.add(ops.H, 0)
    assert result is None
    assert len(p.operations) == 1
    ao = p.operations[0]
    assert isinstance(ao, AppliedOperation)
    assert ao.operation is ops.H
    assert ao.targets == (p.qreg[0][0],)


def test_add_multi_operand_tuple():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    ao = p.operations[0]
    assert ao.targets == (p.qreg[0][0], p.qreg[0][1])


def test_add_parametric_gate():
    p = Program(1)
    p.add(ops.RX(0.2), 0)
    assert p.operations[0].operation.theta == 0.2


def test_add_rejects_variadic_positional():
    p = Program(2)
    with pytest.raises(TypeError):
        p.add(ops.CZ, 0, 1)  # variadic not supported


def test_add_wrong_arity_raises():
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(ops.CZ, 0)  # CZ needs 2 targets


def test_add_rejects_non_operation():
    p = Program(1)
    with pytest.raises(TypeError):
        p.add(ops.RX, 0)  # passed the class, not an instance


def test_add_rejects_int_target_with_multiple_quantum_registers():
    qr0 = QuantumRegister(1, name="a")
    qr1 = QuantumRegister(1, name="b")
    p = Program([qr0, qr1])

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0)


def test_add_rejects_duplicate_qubit_targets():
    p = Program(2)
    with pytest.raises(ValueError, match="more than once"):
        p.add(ops.CZ, (0, 0))


def test_add_accepts_explicit_refs_across_multiple_quantum_registers():
    qr0 = QuantumRegister(1, name="a")
    qr1 = QuantumRegister(1, name="b")
    p = Program([qr0, qr1])

    p.add(ops.X, qr0[0])
    p.add(ops.X, qr1[0])

    assert p.operations[0].targets == (qr0[0],)
    assert p.operations[1].targets == (qr1[0],)


def test_add_swap_levels_out_of_range_raises():
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    with pytest.raises(ValueError, match="0 <= j, k < dim"):
        p.add(ops.SwapLevels(0, 5), 0)


def test_add_swap_levels_in_range_succeeds():
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    p.add(ops.SwapLevels(0, 2), 0)
    assert p.operations[0].operation.j == 0


@pytest.mark.parametrize("op_cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_add_subspace_rotation_out_of_range_raises(op_cls):
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    with pytest.raises(ValueError, match="0 <= j, k < dim"):
        p.add(op_cls(0.3, (1, 3)), 0)


@pytest.mark.parametrize("op_cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_add_subspace_rotation_in_range_succeeds(op_cls):
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    p.add(op_cls(0.3, (1, 2)), 0)
    assert p.operations[0].operation.subspace == (1, 2)
