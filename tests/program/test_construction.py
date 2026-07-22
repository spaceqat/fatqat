"""Tests Program construction, register coercion, and reference resolution."""

import pytest

from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister, ClassicalRegister


def test_int_construction_creates_default_registers():
    p = Program(2, 2)
    assert len(p.qreg) == 1 and p.qreg[0].size == 2
    assert len(p.clreg) == 1 and p.clreg[0].size == 2
    assert p.operations == ()


def test_operations_is_read_only_tuple_view():
    p = Program(1)
    assert isinstance(p.operations, tuple)
    with pytest.raises(AttributeError):
        p.operations.append("bad")


def test_operations_tuple_view_is_cached_until_mutation():
    p = Program(2)
    before = p.operations
    assert p.operations is before

    from fatqat import operations as ops

    p.add(ops.H, 0)
    after = p.operations
    assert after is not before
    assert p.operations is after


def test_zero_classical_means_no_classical_register():
    p = Program(2)
    assert len(p.qreg) == 1
    assert p.clreg == ()


def test_list_construction_with_explicit_registers():
    qr = QuantumRegister(3, name="data")
    cr = ClassicalRegister(2, name="ro")
    p = Program([qr], [cr])
    assert p.qreg == (qr,)
    assert p.clreg == (cr,)


def test_register_collections_are_public_read_only_tuples():
    p = Program(1, 1)
    assert isinstance(p.qreg, tuple)
    assert isinstance(p.clreg, tuple)
    with pytest.raises(AttributeError):
        p.qreg.append(QuantumRegister(1))
    with pytest.raises(AttributeError):
        p.clreg.clear()


def test_flat_quantum_ref_resolution_out_of_range_raises():
    p = Program(2)
    with pytest.raises(IndexError):
        p._resolve_quantum_ref(2)


def test_resolve_quantum_ref_rejects_foreign_ref():
    p = Program(2)
    foreign = QuantumRegister(2, name="other")
    with pytest.raises(ValueError):
        p._resolve_quantum_ref(foreign[0])


def test_resolve_quantum_ref_stays_scalar_only_and_rejects_register_view():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    with pytest.raises(TypeError):
        p._resolve_quantum_ref(atoms.row(0))


def test_metadata_defaults_to_empty_dict():
    p = Program(1)
    assert p.metadata == {}


def test_construction_rejects_float_count():
    with pytest.raises(TypeError, match="int count"):
        Program(2.0, 2)


def test_construction_rejects_bool_count():
    with pytest.raises(TypeError, match="int count"):
        Program(True, 1)


def test_construction_rejects_non_register_in_list():
    with pytest.raises(TypeError, match="QuantumRegister instances"):
        Program([1, 2], 0)


def test_construction_rejects_wrong_register_type_in_list():
    cr = ClassicalRegister(1)
    with pytest.raises(TypeError, match="QuantumRegister instances"):
        Program([cr], 0)
