import pytest

from qnsim.program import Program, AppliedOperation
from qnsim import operations as ops
from qnsim.registers import QuantumRegister


def test_add_single_operand_int():
    p = Program(2)
    p.add(ops.H, 0)
    assert len(p.operations) == 1
    ao = p.operations[0]
    assert isinstance(ao, AppliedOperation)
    assert ao.operation is ops.H
    assert ao.targets == (p.qreg[0][0],)


def test_add_returns_none_and_mutates_in_place():
    p = Program(1)
    assert p.add(ops.X, 0) is None
    assert len(p.operations) == 1


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
    p = Program.registers(qreg=[qr0, qr1])

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0)


def test_add_accepts_explicit_refs_across_multiple_quantum_registers():
    qr0 = QuantumRegister(1, name="a")
    qr1 = QuantumRegister(1, name="b")
    p = Program.registers(qreg=[qr0, qr1])

    p.add(ops.X, qr0[0])
    p.add(ops.X, qr1[0])

    assert p.operations[0].targets == (qr0[0],)
    assert p.operations[1].targets == (qr1[0],)
