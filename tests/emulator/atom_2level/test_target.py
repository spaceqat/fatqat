"""Two-level atom bound-target tests."""

from copy import deepcopy
import json
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest

import fatqat as fq
import fatqat.emulator.atom_2level.target as atom2_target
from fatqat._pulse_values import PulseControl
from fatqat.emulator.atom_2level.model import Atom2LevelModel
from fatqat.emulator.atom_2level.target import _Atom2LevelTarget
from fatqat.errors import BackendValidationError
from fatqat.emulator import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="document")
def document_fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _target(document, sites=2, *, interaction_cutoff=2.0, spacing=2.0):
    return _Atom2LevelTarget(
        Atom2LevelModel.from_document(document),
        fq.emulator.AtomArrangement.rectangular(1, sites, spacing),
        interaction_cutoff,
    )


def test_target_owns_claims_dimensions_interactions_and_digit_map(document):
    target = _target(document)
    assert target.local_dimension == 2
    assert target.hilbert_dimension == 4
    assert target.device_labels == (0, 1)
    assert target.reported_digit_map(0) == (0, 1)
    assert len(target.interactions) == 1
    drive = target.bind_control(target.model.control.drive())
    assert drive.device_operands == (0, 1)
    assert drive.claims == target.bind_gate_operands((0, 1)).claims


def test_claims_are_instance_local_while_addresses_are_portable(document):
    first = _target(document)
    second = _target(deepcopy(document))
    assert first.model.control.drive() == second.model.control.drive()
    assert (
        first.bind_control(first.model.control.drive()).claims
        != second.bind_control(first.model.control.drive()).claims
    )


@pytest.mark.parametrize(
    ("program", "match"),
    [
        (fq.Program(1), "exactly one"),
        (fq.Program([fq.QuantumRegister(2, dim=3)]), "dimension-two"),
    ],
)
def test_program_binding_rejects_count_and_dimension(document, program, match):
    with pytest.raises(BackendValidationError, match=match):
        _target(document).bind_program(program)


def test_program_binding_returns_layout_in_device_order(document):
    target = _target(document)
    program = fq.Program(2)
    binding = target.bind_program(program)
    refs = tuple(program.quantum_registers[0][index] for index in range(2))
    assert binding.device_labels_for(refs) == (0, 1)


def test_program_binding_reads_each_declared_resource_once(document):
    register = fq.QuantumRegister(2)
    reads = []

    class CountingRegister:
        size = register.size

        def __getitem__(self, index):
            reads.append(index)
            return register[index]

    binding = _target(document).bind_program(
        SimpleNamespace(quantum_registers=(CountingRegister(),))
    )
    assert len(binding.refs) == 2
    assert reads == [0, 1]


def test_no_cutoff_prepares_every_signed_interaction_once(document):
    target = _target(document, 3, interaction_cutoff=None)
    assert tuple((value.first, value.second) for value in target.interactions) == (
        (0, 1),
        (0, 2),
        (1, 2),
    )


def test_cutoff_selects_coordinate_distance_shells_and_preserves_strength(document):
    document["parameters"]["c6"] = -64.0
    arrangement = fq.emulator.AtomArrangement.rectangular(2, 3, 2.0)
    model = Atom2LevelModel.from_document(document)

    all_pairs = _Atom2LevelTarget(model, arrangement, None)
    no_pairs = _Atom2LevelTarget(model, arrangement, 0.0)
    nearest = _Atom2LevelTarget(model, arrangement, arrangement.spacing)
    diagonals = _Atom2LevelTarget(model, arrangement, sqrt(2) * arrangement.spacing)

    assert len(all_pairs.interactions) == 15
    assert no_pairs.interactions == ()
    assert tuple((item.first, item.second) for item in nearest.interactions) == (
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 5),
    )
    assert len(diagonals.interactions) == 11
    assert nearest.interactions[0].signed_strength_rad_per_us == -1.0


def test_cutoff_boundary_keeps_decimal_nearest_pairs_without_diagonals(document):
    arrangement = fq.emulator.AtomArrangement.rectangular(2, 10, 0.1)
    target = _Atom2LevelTarget(
        Atom2LevelModel.from_document(document), arrangement, arrangement.spacing
    )

    expected_pairs = 2 * 9 + 10
    assert len(target.interactions) == expected_pairs
    assert all(item.distance_um < 0.11 for item in target.interactions)


def test_cutoff_selection_uses_only_coordinates(document):
    arrangement = SimpleNamespace(
        num_sites=3,
        coordinates=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    target = _Atom2LevelTarget(
        Atom2LevelModel.from_document(document), arrangement, 1.0
    )

    assert tuple((item.first, item.second) for item in target.interactions) == ((0, 1),)


def test_target_owns_duration_and_complete_interpolant_limits(document):
    duration_document = deepcopy(document)
    limits = duration_document["parameters"]["channel_limits"]["rydberg_global"]
    limits.update(min_duration=0.5, max_duration=2.0)
    duration_target = _target(duration_document)
    for duration, message in ((0.25, "below"), (3.0, "exceeds")):
        control = PulseControl(
            duration_target.model.control.drive(),
            SampledWaveform((0.0, duration), (0.0, 0.0)),
        )
        binding = duration_target.bind_control(control.channel)
        with pytest.raises(BackendValidationError, match=message):
            duration_target.validate_pulse_controls((control,), (binding,), duration)

    amplitude_document = deepcopy(document)
    amplitude_document["parameters"]["channel_limits"]["rydberg_global"][
        "max_amplitude"
    ] = 1.0
    amplitude_target = _target(amplitude_document)
    control = PulseControl(
        amplitude_target.model.control.drive(),
        SampledWaveform(
            (0.0, 1.0, 2.0, 3.0),
            (0.0, 1.0j, 1.0j, 0.0),
        ),
    )
    binding = amplitude_target.bind_control(control.channel)
    with pytest.raises(BackendValidationError, match="drive magnitude.*exceeds"):
        amplitude_target.validate_pulse_controls((control,), (binding,), 3.0)


def test_target_owns_real_detuning_and_signed_cubic_extrema(document):
    bounded_document = deepcopy(document)
    limits = bounded_document["parameters"]["channel_limits"]["rydberg_global"]
    limits.update(min_detuning=-1.0, max_detuning=1.0)
    target = _target(bounded_document)

    complex_control = PulseControl(
        target.model.control.detuning(),
        SampledWaveform((0.0, 1.0), (0.1j, 0.1j)),
    )
    binding = target.bind_control(complex_control.channel)
    with pytest.raises(BackendValidationError, match="must be real"):
        target.validate_pulse_controls((complex_control,), (binding,), 1.0)

    overshooting = PulseControl(
        target.model.control.detuning(),
        SampledWaveform(
            (0.0, 1.0, 2.0, 3.0),
            (0.0, 1.0, 1.0, 0.0),
        ),
    )
    binding = target.bind_control(overshooting.channel)
    with pytest.raises(BackendValidationError, match="detuning.*exceeds"):
        target.validate_pulse_controls((overshooting,), (binding,), 3.0)


def test_real_detuning_skips_spline_analysis_without_limits(document, monkeypatch):
    target = _target(document)
    control = PulseControl(
        target.model.control.detuning(),
        SampledWaveform((0.0, 1.0), (-0.2, 0.3)),
    )
    binding = target.bind_control(control.channel)

    def spline_analysis_must_not_run(*_args):
        raise AssertionError("detuning spline analysis ran without limits")

    monkeypatch.setattr(
        atom2_target,
        "_real_spline_minimum_and_maximum",
        spline_analysis_must_not_run,
    )

    target.validate_pulse_controls((control,), (binding,), 1.0)
