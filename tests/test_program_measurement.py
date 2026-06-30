import pytest

from qnsim.program import Program, Measurement
from qnsim import operations as ops
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_add_measurement_appends_measurement():
    p = Program(2, 2)
    p.add_measurement(0, 0)
    assert len(p.operations) == 1
    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.qreg == p.qreg[0][0]
    assert m.clreg == p.creg[0][0]


def test_add_measurement_returns_none():
    p = Program(1, 1)
    assert p.add_measurement(0, 0) is None


def test_operations_preserve_order_and_type_mix():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    assert len(p.operations) == 4
    assert p.operations[0].operation.name == "H"
    assert isinstance(p.operations[2], Measurement)


def test_add_measurement_rejects_quantum_ref_as_clreg():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add_measurement(0, p.qreg[0][1])  # quantum ref as classical slot


def test_add_measurement_int_clreg_rejects_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add_measurement(0, 0)


def test_add_measurement_explicit_clreg_ref_works_with_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    p.add_measurement(1, p.creg[1][0])

    assert p.operations[0].qreg == p.qreg[0][1]
    assert p.operations[0].clreg == p.creg[1][0]


def test_measurement_metadata_is_copied_not_aliased():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    meta = {"k": 1}
    m = Measurement(qreg=qr[0], clreg=cr[0], metadata=meta)
    meta["k"] = 2
    assert m.metadata == {"k": 1}
