"""Superconducting transmon bound-target tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import fatqat as fq
from fatqat.emulator.superconducting.model import TransmonModel
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.errors import BackendValidationError


def test_target_precomputes_topology_and_semantic_bindings(model_document):
    target = _TransmonTarget(TransmonModel.from_document(model_document))
    drive = target.bind_control(target.model.control.drive("q0"))
    exchange = target.bind_control(target.model.control.exchange("q0", "q1"))
    frame = target.bind_frame(target.model.frame("q1"))
    assert target.device_labels == ("q0", "q1")
    assert (target.local_dimension, target.hilbert_dimension) == (3, 9)
    assert drive.kind == "drive"
    assert drive.device_operands == ("q0",)
    assert (
        exchange.kind == "exchange"
        and exchange.allows_additional_claims
        and exchange.device_operands == ("q0", "q1")
    )
    assert len(exchange.claims) == 3
    assert frame.device_operands == ("q1",)
    assert target.reported_digit_map("q0") == (0, 1, 1)
    with pytest.raises(TypeError):
        target._subsystem_ordinals["q2"] = 2
    with pytest.raises(TypeError):
        target._coupling_ordinals[frozenset(("q0", "q2"))] = 1


def test_addresses_are_portable_but_claims_are_target_local(model_document):
    first = _TransmonTarget(TransmonModel.from_document(model_document))
    second = _TransmonTarget(TransmonModel.from_document(deepcopy(model_document)))
    address = second.model.control.drive("q0")
    assert address == first.model.control.drive("q0")
    assert first.bind_control(address).claims != second.bind_control(address).claims


def test_program_binding_accepts_only_a_binary_declaration_prefix(model_document):
    target = _TransmonTarget(TransmonModel.from_document(model_document))
    program = fq.Program(1)
    binding = target.bind_program(program)
    assert binding.device_label(program.quantum_registers[0][0]) == "q0"
    with pytest.raises(BackendValidationError, match="requires 3"):
        target.bind_program(fq.Program(3))
    with pytest.raises(BackendValidationError, match="dimension-two"):
        target.bind_program(fq.Program([fq.QuantumRegister(1, dim=3)]))


def test_program_binding_reads_each_declared_resource_once(model_document):
    register = fq.QuantumRegister(2)
    reads = []

    class CountingRegister:
        size = register.size

        def __getitem__(self, index):
            reads.append(index)
            return register[index]

    binding = _TransmonTarget(TransmonModel.from_document(model_document)).bind_program(
        SimpleNamespace(quantum_registers=(CountingRegister(),))
    )
    assert len(binding.refs) == 2
    assert reads == [0, 1]


def test_target_rejects_unknown_labels_edges_and_families(
    model_document, atom_3level_model
):
    target = _TransmonTarget(TransmonModel.from_document(model_document))
    with pytest.raises(BackendValidationError, match="unknown model subsystem"):
        target.bind_control(target.model.control.drive("missing"))
    with pytest.raises(BackendValidationError, match="no declared coupling"):
        target.bind_control(target.model.control.exchange("q0", "missing"))
    with pytest.raises(BackendValidationError, match="foreign"):
        target.bind_control(atom_3level_model.control.raman(0))
