"""Tests for the public ResourceLayout: RegisterRef -> device resource label."""

import pytest

from fatqat._engine_index_allocation import _EngineIndexAllocation
from fatqat.backends import SimulatorBackend
from fatqat.program import Program
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


def test_device_labels_for_preserves_operand_order():
    qreg = QuantumRegister(3, name="q")
    layout = ResourceLayout({qreg[0]: "a", qreg[1]: "b", qreg[2]: "c"})
    assert layout.device_labels_for((qreg[2], qreg[0])) == ("c", "a")
    assert layout.device_labels_for(()) == ()


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


def test_foreign_ref_raises_on_device_labels_for():
    qreg = QuantumRegister(1, name="q")
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout({qreg[0]: 0})
    with pytest.raises(KeyError):
        layout.device_labels_for((qreg[0], foreign[0]))


def test_lookalike_register_is_not_part_of_the_layout():
    qreg = QuantumRegister(1, name="q")
    layout = ResourceLayout({qreg[0]: 0})
    lookalike = QuantumRegister(qreg.size, name=qreg.name)
    with pytest.raises(KeyError):
        layout.device_label(lookalike[0])


# --- SimulatorBackend's default resource-mapping hook -----------------------


def test_simulator_backend_resolves_generic_identity_device_labels_in_declaration_order():
    program = Program(3)
    backend = SimulatorBackend()

    resource_layout = backend._resolve_resource_layout(program)

    refs = [program.quantum_registers[0][i] for i in range(3)]
    assert [resource_layout.device_label(ref) for ref in refs] == [0, 1, 2]
    assert isinstance(resource_layout, ResourceLayout)


def test_simulator_backend_allocate_engine_indices_returns_a_separate_engine_index_allocation():
    program = Program(3)
    backend = SimulatorBackend()

    resource_layout = backend._resolve_resource_layout(program)
    engine_index_allocation = backend._allocate_engine_indices(program)

    refs = [program.quantum_registers[0][i] for i in range(3)]
    # Two independently-resolved values: not the same object...
    assert resource_layout is not engine_index_allocation
    assert isinstance(engine_index_allocation, _EngineIndexAllocation)
    # ...but the generic simulator's trivial policy makes their current
    # numerical values coincide. That coincidence is not an API contract.
    assert [engine_index_allocation.subsystem_index(ref) for ref in refs] == [
        resource_layout.device_label(ref) for ref in refs
    ]
