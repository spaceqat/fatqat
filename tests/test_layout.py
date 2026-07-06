"""Tests resource layout flattening and reference lookup."""

import pytest

from fatqcat.layout import ResourceLayout
from fatqcat.program import Program
from fatqcat.registers import QuantumRegister, ClassicalRegister


def test_single_register_layout():
    p = Program(3, 2)
    layout = ResourceLayout.from_program(p)
    assert layout.system_dims == (2, 2, 2)
    assert layout.n_subsystems == 3
    assert layout.n_clbits == 2
    assert layout.subsystem_index(p.qreg[0][0]) == 0
    assert layout.subsystem_index(p.qreg[0][2]) == 2
    assert layout.clbit_index(p.creg[0][1]) == 1


def test_multi_register_flat_concatenation():
    qa = QuantumRegister(2, name="a")
    qb = QuantumRegister(2, name="b")
    p = Program([qa, qb])
    layout = ResourceLayout.from_program(p)
    assert layout.subsystem_index(qa[0]) == 0
    assert layout.subsystem_index(qa[1]) == 1
    assert layout.subsystem_index(qb[0]) == 2
    assert layout.subsystem_index(qb[1]) == 3
    assert layout.system_dims == (2, 2, 2, 2)


def test_unknown_ref_raises():
    p = Program(1)
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout.from_program(p)
    with pytest.raises(KeyError):
        layout.subsystem_index(foreign[0])


def test_heterogeneous_system_dims():
    qt = QuantumRegister(3, dim=3, name="t")
    qb = QuantumRegister(2, dim=2, name="b")
    p = Program([qt, qb], [ClassicalRegister(3, dim=3), ClassicalRegister(2, dim=2)])
    layout = ResourceLayout.from_program(p)
    assert layout.system_dims == (3, 3, 3, 2, 2)
    assert layout.classical_dims == (3, 3, 3, 2, 2)


def test_classical_dims_default_binary():
    p = Program(2, 2)
    layout = ResourceLayout.from_program(p)
    assert layout.classical_dims == (2, 2)
