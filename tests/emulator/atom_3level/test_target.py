"""Three-level atom bound-target tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import fatqat as fq
from fatqat.emulator.atom_3level.model import Atom3LevelModel
from fatqat.emulator.atom_3level.target import _Atom3LevelTarget
from fatqat.errors import BackendValidationError


def _target(document, sites=2):
    return _Atom3LevelTarget(
        Atom3LevelModel.from_document(document),
        fq.AtomArrangement.rectangular(1, sites, 2.0),
    )


def test_target_binds_controls_frames_claims_dimensions_and_interactions(
    atom_3level_model_document,
):
    target = _target(atom_3level_model_document)
    control = target.bind_control(target.model.control.raman(1))
    frame = target.bind_frame(target.model.frame(1))
    assert (target.local_dimension, target.hilbert_dimension) == (3, 9)
    assert target.device_labels == (0, 1)
    assert target.reported_digit_map(1) == (0, 1, 1)
    assert control.kind == "raman_01"
    assert control.device_operands == (1,)
    assert control.claims == frame.claims
    assert len(target.interactions) == 1
    assert target.interactions[0].signed_strength_rad_per_us == pytest.approx(
        target.model.c6_angular_per_us_um6 / 2**6
    )


def test_addresses_are_portable_but_claims_are_target_local(atom_3level_model_document):
    first = _target(atom_3level_model_document)
    second = _target(deepcopy(atom_3level_model_document))
    assert first.model.control.raman(0) == second.model.control.raman(0)
    assert (
        first.bind_control(second.model.control.raman(0)).claims
        != second.bind_control(second.model.control.raman(0)).claims
    )


def test_program_binding_requires_every_binary_arrangement_site(
    atom_3level_model_document,
):
    target = _target(atom_3level_model_document)
    with pytest.raises(BackendValidationError, match="exactly one"):
        target.bind_program(fq.Program(1))
    with pytest.raises(BackendValidationError, match="dimension-two"):
        target.bind_program(fq.Program([fq.QuantumRegister(2, dim=3)]))
    program = fq.Program(2)
    binding = target.bind_program(program)
    refs = tuple(program.quantum_registers[0][index] for index in range(2))
    assert binding.device_labels_for(refs) == (0, 1)


def test_program_binding_reads_each_declared_resource_once(atom_3level_model_document):
    register = fq.QuantumRegister(2)
    reads = []

    class CountingRegister:
        size = register.size

        def __getitem__(self, index):
            reads.append(index)
            return register[index]

    binding = _target(atom_3level_model_document).bind_program(
        SimpleNamespace(quantum_registers=(CountingRegister(),))
    )
    assert len(binding.refs) == 2
    assert reads == [0, 1]


def test_target_has_no_runtime_binding_or_interaction_cache_shapes(
    atom_3level_model_document,
):
    target = _target(atom_3level_model_document, 3)
    for removed in (
        "binding_snapshot",
        "occupancy",
        "logical_to_site",
        "interaction_provider",
        "interaction_cache",
    ):
        assert not hasattr(target, removed)
    assert len(target.interactions) == 3
