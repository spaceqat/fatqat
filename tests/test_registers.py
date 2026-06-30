import pytest

from qnsim.registers import (
    Register,
    QuantumRegister,
    ClassicalRegister,
    RegisterRef,
)


def test_getitem_returns_registerref():
    qr = QuantumRegister(3, name="q")
    ref = qr[1]
    assert isinstance(ref, RegisterRef)
    assert ref.register is qr
    assert ref.index == 1


def test_getitem_out_of_range_raises_indexerror():
    qr = QuantumRegister(2)
    with pytest.raises(IndexError):
        qr[2]
    with pytest.raises(IndexError):
        qr[-1]


def test_size_first_construction_and_keyword_name():
    cr = ClassicalRegister(4, name="c")
    assert cr.size == 4
    assert cr.name == "c"


def test_non_positive_size_rejected():
    with pytest.raises(ValueError):
        QuantumRegister(0)
    with pytest.raises(ValueError):
        ClassicalRegister(-1)


def test_registers_are_frozen():
    qr = QuantumRegister(1)
    with pytest.raises(Exception):
        qr.size = 5


def test_register_metadata_is_copied_not_aliased():
    meta = {"k": 1}
    qr = QuantumRegister(1, metadata=meta)
    meta["k"] = 2
    assert qr.metadata == {"k": 1}
