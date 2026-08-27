"""Custom portable pulse-map workflows through the public transmon backend."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator import (
    PhaseShift,
    PulseControl,
    PulseDefinition,
    PulseImplementationMap,
    TransmonEmulator,
    default_transmon_gate_implementation_map,
)
from fatqat.errors import BackendValidationError, PulseImplementationError
from fatqat.emulator import SampledWaveform


def _drive_definition(model, subsystem_id, amplitude):
    return PulseDefinition(
        1.0,
        (
            PulseControl(
                model.control.drive(subsystem_id),
                SampledWaveform((0.0, 1.0), (amplitude, amplitude)),
            ),
        ),
    )


def _defaults(model, calibration):
    return default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )


def test_device_specific_table_dispatches_by_ordered_operands(model):
    implementations = PulseImplementationMap()
    implementations.add(
        ops.CZ,
        _drive_definition(model, "q0", 0.11),
        device_operands=("q0", "q1"),
    )
    implementations.add(
        ops.CZ,
        _drive_definition(model, "q0", 0.22),
        device_operands=("q1", "q0"),
    )
    backend = TransmonEmulator(model, gate_implementation_map=implementations)

    for targets, amplitude in (((0, 1), 0.11), ((1, 0), 0.22)):
        program = fq.Program(2)
        program.add(ops.CZ, targets)
        (block,) = backend._prepare_program(program).plan
        assert np.allclose(block.controls[0].waveform.values, amplitude)


def test_non_definition_and_arbitrary_rule_failures_use_public_error_boundary(
    model, calibration
):
    for implementation, expected in (
        (lambda operation: None, PulseImplementationError),
        (
            lambda operation: (_ for _ in ()).throw(ValueError("bad recipe")),
            PulseImplementationError,
        ),
    ):
        implementations = _defaults(model, calibration)
        implementations.remove(ops.CZ)
        implementations.add(
            ops.CZ,
            implementation,
            device_operands=("q0", "q1"),
        )
        backend = TransmonEmulator(model, gate_implementation_map=implementations)
        program = fq.Program(2)
        program.add(ops.CZ, (0, 1))
        with pytest.raises(expected):
            backend.run(program)


def test_structural_control_from_distinct_source_binds_to_target(
    model, build_model_and_calibration
):
    other, _ = build_model_and_calibration()
    implementations = PulseImplementationMap()
    implementations.add(
        ops.X,
        _drive_definition(other, "q0", 0.0),
        device_operands=("q0",),
    )
    backend = TransmonEmulator(model, gate_implementation_map=implementations)
    program = fq.Program(1)
    program.add(ops.X, 0)
    (block,) = backend._prepare_program(program).plan
    assert block.controls[0].channel == model.control.drive("q0")


def test_definition_shape_and_target_containment_fail_at_owned_boundaries(
    model, calibration
):
    with pytest.raises(BackendValidationError, match="extends"):
        PulseDefinition(
            1.0,
            (
                PulseControl(
                    model.control.drive("q0"),
                    SampledWaveform((0.0, 2.0), (0.0, 0.0)),
                ),
            ),
        )

    implementations = _defaults(model, calibration)
    implementations.remove(ops.RX)
    implementations.add(
        ops.RX,
        _drive_definition(model, "q1", 0.0),
        device_operands=("q0",),
    )
    backend = TransmonEmulator(model, gate_implementation_map=implementations)
    program = fq.Program(1)
    program.add(ops.RX(0.2), 0)
    with pytest.raises(BackendValidationError, match="outside"):
        backend.run(program)


def test_documented_custom_cz_uses_only_public_structural_values(model, calibration):
    def custom_cz(operation, *, device_operands):
        del operation
        first, second = device_operands
        duration = 10.0
        return PulseDefinition(
            duration,
            (
                PulseControl(
                    model.control.exchange(first, second),
                    SampledWaveform((0.0, duration), (0.0, 0.0)),
                ),
            ),
            (PhaseShift(model.frame(first), 0.05),),
        )

    implementations = _defaults(model, calibration)
    implementations.remove(ops.CZ)
    implementations.add(ops.CZ, custom_cz)
    backend = fq.emulator.TransmonEmulator(
        model, gate_implementation_map=implementations
    )
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    (block,) = backend._prepare_program(program).plan
    assert block.control_bindings[0].engine_indices == (0, 1)
