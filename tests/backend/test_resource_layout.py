"""Tests for the public ResourceLayout: RegisterRef -> device resource label."""

import pytest

from fatqat.registers import QuantumRegister
from fatqat.resource_layout import ResourceLayout


def test_device_label_lookup():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: 0, qreg[1]: 1})
    assert layout.device_label(qreg[0]) == 0
    assert layout.device_label(qreg[1]) == 1


def test_device_label_opaque_hashable_values():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: "site-A", qreg[1]: (3, 7)})
    assert layout.device_label(qreg[0]) == "site-A"
    assert layout.device_label(qreg[1]) == (3, 7)


def test_device_operands_preserves_operand_order():
    qreg = QuantumRegister(3, name="q")
    layout = ResourceLayout({qreg[0]: "a", qreg[1]: "b", qreg[2]: "c"})
    assert layout.device_operands((qreg[2], qreg[0])) == ("c", "a")
    assert layout.device_operands(()) == ()


def test_device_labels_membership():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: 5, qreg[1]: 9})
    assert layout.device_labels == frozenset({5, 9})


def test_foreign_ref_raises_on_device_label():
    qreg = QuantumRegister(1, name="q")
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout({qreg[0]: 0})
    with pytest.raises(KeyError):
        layout.device_label(foreign[0])


def test_foreign_ref_raises_on_device_operands():
    qreg = QuantumRegister(1, name="q")
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout({qreg[0]: 0})
    with pytest.raises(KeyError):
        layout.device_operands((qreg[0], foreign[0]))


def test_lookalike_register_is_not_part_of_the_layout():
    qreg = QuantumRegister(1, name="q")
    layout = ResourceLayout({qreg[0]: 0})
    lookalike = QuantumRegister(qreg.size, name=qreg.name)
    with pytest.raises(KeyError):
        layout.device_label(lookalike[0])
