import pytest

from qnsim.program import Program
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_int_construction_creates_default_registers():
    p = Program(2, 2)
    assert len(p.qreg) == 1 and p.qreg[0].size == 2
    assert len(p.creg) == 1 and p.creg[0].size == 2
    assert p.operations == []


def test_zero_classical_means_no_classical_register():
    p = Program(2)
    assert len(p.qreg) == 1
    assert p.creg == []


def test_registers_classmethod_with_explicit_registers():
    qr = QuantumRegister(3, name="data")
    cr = ClassicalRegister(2, name="ro")
    p = Program.registers(qreg=[qr], clreg=[cr])
    assert p.qreg == [qr]
    assert p.creg == [cr]


def test_flat_qubit_resolution_across_registers():
    qr0 = QuantumRegister(2, name="a")
    qr1 = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qr0, qr1])
    assert p._resolve_qubit(0) == qr0[0]
    assert p._resolve_qubit(1) == qr0[1]
    assert p._resolve_qubit(2) == qr1[0]
    assert p._resolve_qubit(3) == qr1[1]


def test_flat_qubit_resolution_out_of_range_raises():
    p = Program(2)
    with pytest.raises(IndexError):
        p._resolve_qubit(2)


def test_resolve_qubit_rejects_foreign_ref():
    p = Program(2)
    foreign = QuantumRegister(2, name="other")
    with pytest.raises(ValueError):
        p._resolve_qubit(foreign[0])


def test_metadata_defaults_to_empty_dict():
    p = Program(1)
    assert p.metadata == {}
