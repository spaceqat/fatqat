import pytest

from qnsim.program import Program, Measurement
from qnsim import operations as ops


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
