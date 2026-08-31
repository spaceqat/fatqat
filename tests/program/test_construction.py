"""Tests Program construction, register coercion, and reference resolution."""

import pytest

from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister, ClassicalRegister


def test_int_construction_creates_default_registers():
    p = Program(2, 2)
    assert len(p.quantum_registers) == 1 and p.quantum_registers[0].size == 2
    assert len(p.classical_registers) == 1 and p.classical_registers[0].size == 2
    assert p._instructions == ()


def test_long_register_names_are_canonical():
    p = Program(quantum_registers=2, classical_registers=1)
    assert len(p.quantum_registers) == 1
    assert len(p.classical_registers) == 1


def test_instructions_is_read_only_tuple_view():
    p = Program(1)
    assert isinstance(p._instructions, tuple)
    with pytest.raises(AttributeError):
        p._instructions.append("bad")


def test_instructions_tuple_view_is_cached_until_mutation():
    p = Program(2)
    before = p._instructions
    assert p._instructions is before

    import fatqat.operations as ops

    p.add(ops.H, 0)
    after = p._instructions
    assert after is not before
    assert p._instructions is after


def test_zero_classical_means_no_classical_register():
    p = Program(2)
    assert len(p.quantum_registers) == 1
    assert p.classical_registers == ()


def test_list_construction_with_explicit_registers():
    qr = QuantumRegister(3, name="data")
    cr = ClassicalRegister(2, name="ro")
    p = Program([qr], [cr])
    assert p.quantum_registers == (qr,)
    assert p.classical_registers == (cr,)


def test_tuple_construction_with_explicit_registers():
    qr = QuantumRegister(3, name="data")
    cr = ClassicalRegister(2, name="ro")
    p = Program((qr,), (cr,))
    assert p.quantum_registers == (qr,)
    assert p.classical_registers == (cr,)


def test_register_collections_are_public_read_only_tuples():
    p = Program(1, 1)
    assert isinstance(p.quantum_registers, tuple)
    assert isinstance(p.classical_registers, tuple)
    with pytest.raises(AttributeError):
        p.quantum_registers.append(QuantumRegister(1))
    with pytest.raises(AttributeError):
        p.classical_registers.clear()


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
    qubits = GridRegister(2, 2, name="qubits")
    p = Program([qubits])
    with pytest.raises(TypeError):
        p._resolve_quantum_ref(qubits.row(0))


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


def test_duplicate_register_object_rejected():
    qreg = QuantumRegister(2, name="q")
    with pytest.raises(ValueError, match="more than once"):
        Program([qreg, qreg])

    creg = ClassicalRegister(2, name="c")
    with pytest.raises(ValueError, match="more than once"):
        Program(2, [creg, creg])


def test_distinct_equal_registers_accepted():
    program = Program([QuantumRegister(1, name="q"), QuantumRegister(1, name="q")])
    assert len(program.quantum_registers) == 2


def test_falsy_non_mapping_metadata_rejected_like_register():
    with pytest.raises(TypeError):
        Program(1, metadata=0)
    assert Program(1, metadata={}).metadata == {}
    assert Program(1, metadata=None).metadata == {}
