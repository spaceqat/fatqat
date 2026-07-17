"""Tests register construction, indexing, metadata copying, and immutability."""

import pytest

from fatqat.registers import (
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


def test_dim_defaults_to_two():
    assert QuantumRegister(3).dim == 2
    assert ClassicalRegister(2).dim == 2


def test_dim_can_be_set():
    qr = QuantumRegister(3, dim=3)
    assert qr.dim == 3
    assert qr[0].register.dim == 3


def test_dim_below_two_rejected():
    with pytest.raises(ValueError):
        QuantumRegister(2, dim=1)
    with pytest.raises(ValueError):
        QuantumRegister(2, dim=0)
    with pytest.raises(ValueError):
        ClassicalRegister(2, dim=-3)


def test_dim_non_int_rejected():
    with pytest.raises(TypeError):
        QuantumRegister(2, dim=2.0)
    with pytest.raises(TypeError):
        QuantumRegister(2, dim=True)


def test_distinct_registers_with_identical_fields_are_not_equal():
    assert QuantumRegister(2, name="q") != QuantumRegister(2, name="q")
    assert ClassicalRegister(2, name="c") != ClassicalRegister(2, name="c")


def test_registers_are_hashable_despite_metadata():
    assert isinstance(hash(QuantumRegister(1, metadata={"k": 1})), int)
    assert isinstance(hash(ClassicalRegister(1, metadata={"k": 1})), int)


def test_distinct_registers_do_not_collapse_in_a_set():
    assert len({QuantumRegister(1), QuantumRegister(1)}) == 2


def test_registerref_compares_by_register_identity_and_index():
    a = QuantumRegister(2, name="q")
    b = QuantumRegister(2, name="q")
    assert a[0] == a[0]
    assert a[0] != a[1]
    assert a[0] != b[0]


def test_registerref_is_usable_as_a_dict_key():
    a = QuantumRegister(2, name="q")
    b = QuantumRegister(2, name="q")
    offsets = {a[0]: "a0", b[0]: "b0"}
    assert offsets[a[0]] == "a0"
    assert offsets[b[0]] == "b0"
    assert len(offsets) == 2
