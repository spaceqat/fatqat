"""Direct global two-level control lowering and validation contracts."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
from fatqat._pulse_values import ControlChannel, PulseControl
from fatqat.emulator.atom_2level import Atom2LevelEmulator, Atom2LevelModel
from fatqat.errors import BackendValidationError
from fatqat.waveforms import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@dataclass(frozen=True)
class _ForeignControl(ControlChannel):
    pass


def _model(**limits):
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document["parameters"]["channel_limits"]["rydberg_global"].update(limits)
    return Atom2LevelModel.from_document(document)


def _backend(model, *, rows=1, columns=2, spacing=2.0):
    return Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(rows, columns, spacing),
    )


def _waveform(duration, values, times=None):
    selected_times = (0.0, duration) if times is None else times
    selected_values = (values, values) if np.isscalar(values) else values
    return SampledWaveform(selected_times, selected_values)


def _control(channel, duration, values, *, times=None, start_offset=0.0):
    return PulseControl(
        channel,
        _waveform(duration, values, times),
        start_offset=start_offset,
    )


def _lower(backend, operation):
    program = fq.Program(backend.arrangement.num_sites)
    program.add(operation)
    return backend._prepare_program(program).plan[0]


@pytest.mark.parametrize("kind", ["drive", "detuning"])
def test_drive_only_and_detuning_only_lower_as_global_controls(kind):
    model = _model()
    backend = _backend(model)
    channel = model.control.drive() if kind == "drive" else model.control.detuning()
    operation = fq.ops.PulseOperation(2.0, (_control(channel, 2.0, 0.3j),))
    if kind == "detuning":
        operation = fq.ops.PulseOperation(2.0, (_control(channel, 2.0, -0.3),))

    block = _lower(backend, operation)

    assert block.duration == 2.0
    assert block.controls == operation.controls
    assert block.resource_claims == backend._target._claims
    assert block.target_indices == (0, 1)


def test_complex_drive_and_real_detuning_keep_knots_offsets_and_phase():
    model = _model()
    backend = _backend(model)
    drive = PulseControl(
        model.control.drive(),
        SampledWaveform(
            (0.0, 0.2, 0.8),
            (0.2, 0.3j, -0.1 + 0.2j),
        ),
        start_offset=0.2,
    )
    detuning = PulseControl(
        model.control.detuning(),
        SampledWaveform((0.0, 0.4, 1.0), (-1.0, 0.0, 2.0)),
    )
    operation = fq.ops.PulseOperation(1.0, (drive, detuning))

    block = _lower(backend, operation)

    assert block.controls == (drive, detuning)
    assert block.controls[0].waveform.values == drive.waveform.values
    assert block.controls[0].start_offset == 0.2
    assert block.controls[1].waveform.values == detuning.waveform.values


def test_sequential_direct_operations_remain_distinct_blocks():
    model = _model()
    backend = _backend(model)
    operations = (
        fq.ops.PulseOperation(0.4, (_control(model.control.drive(), 0.4, 0.1 + 0.2j),)),
        fq.ops.PulseOperation(0.5, (_control(model.control.detuning(), 0.5, -0.3),)),
        fq.ops.PulseOperation(0.6, (_control(model.control.drive(), 0.6, -0.2j),)),
    )
    program = fq.Program(2)
    for operation in operations:
        program.add(operation)

    plan = backend._prepare_program(program).plan

    assert tuple(block.controls for block in plan) == tuple(
        operation.controls for operation in operations
    )


def test_duplicate_controls_and_endpoint_overflow_are_rejected_by_values():
    model = _model()
    first = _control(model.control.drive(), 1.0, 0.1)
    with pytest.raises(ValueError, match="one channel"):
        fq.ops.PulseOperation(1.0, (first, first))
    with pytest.raises(ValueError, match="extends beyond"):
        fq.ops.PulseOperation(
            1.0,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, 0.8), (0.0, 0.1)),
                    start_offset=0.3,
                ),
            ),
        )


def test_duration_and_complete_interpolant_limits_are_enforced():
    duration_model = _model(min_duration=0.5, max_duration=2.0)
    duration_backend = _backend(duration_model)
    for duration, message in ((0.25, "below"), (3.0, "exceeds")):
        operation = fq.ops.PulseOperation(
            duration,
            (_control(duration_model.control.drive(), duration, 0.0),),
        )
        with pytest.raises(BackendValidationError, match=message):
            _lower(duration_backend, operation)

    bounded_model = _model(max_amplitude=1.0)
    bounded_backend = _backend(bounded_model)
    adversarial = SampledWaveform(
        (0.0, 1.0, 2.0, 3.0),
        (0.0, 1.0j, 1.0j, 0.0),
    )
    operation = fq.ops.PulseOperation(
        3.0, (PulseControl(bounded_model.control.drive(), adversarial),)
    )
    with pytest.raises(BackendValidationError, match="drive magnitude.*exceeds"):
        _lower(bounded_backend, operation)


def test_detuning_is_real_and_uses_signed_cubic_extrema():
    model = _model(min_detuning=-1.0, max_detuning=1.0)
    backend = _backend(model)
    complex_operation = fq.ops.PulseOperation(
        1.0, (_control(model.control.detuning(), 1.0, 0.1j),)
    )
    with pytest.raises(BackendValidationError, match="must be real"):
        _lower(backend, complex_operation)

    overshooting = SampledWaveform(
        (0.0, 1.0, 2.0, 3.0),
        (0.0, 1.0, 1.0, 0.0),
    )
    operation = fq.ops.PulseOperation(
        3.0, (PulseControl(model.control.detuning(), overshooting),)
    )
    with pytest.raises(BackendValidationError, match="detuning.*exceeds"):
        _lower(backend, operation)


def test_structural_controls_reuse_across_models_and_arrangements():
    first_model = _model()
    second_document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    second_document["parameters"]["c6"] = -2.0
    second_model = Atom2LevelModel.from_document(deepcopy(second_document))
    operation = fq.ops.PulseOperation(
        0.4, (_control(first_model.control.drive(), 0.4, 0.2j),)
    )

    first_block = _lower(_backend(first_model), operation)
    second_block = _lower(
        _backend(second_model, rows=2, columns=1, spacing=3.0), operation
    )

    assert first_model.control.drive() == second_model.control.drive()
    assert first_model.control.detuning() == second_model.control.detuning()
    assert first_block.controls == second_block.controls == operation.controls


def test_wrong_family_control_is_rejected():
    model = _model()
    backend = _backend(model)
    program = fq.Program(2)
    operation = fq.ops.PulseOperation(0.4, (_control(_ForeignControl(), 0.4, 0.1),))
    program.add(operation)
    with pytest.raises(BackendValidationError, match="foreign two-level atom control"):
        backend._prepare_program(program)
