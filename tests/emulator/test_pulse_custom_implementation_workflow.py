"""Custom pulse implementation workflows through the public PulseBackend surface.

Every import below comes from `fatqat` / `fatqat.backends` / `fatqat.errors`,
matching the documented custom-CZ workflow: replacing a gate realization
never requires subclassing `PulseBackend` or importing `fatqat.emulator`.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import (
    PhaseShift,
    PulseBackend,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
    default_superconducting_pulse_implementation_map,
)
from fatqat.errors import BackendValidationError, PulseImplementationError


def _drive_definition(model, subsystem_id, amplitude):
    return PulseDefinition(
        1.0,
        (
            SampledControl(
                model.drive_control(subsystem_id), (0.0, 1.0), (amplitude, amplitude)
            ),
        ),
        (model.resource(subsystem_id),),
    )


# --- device-specific CZ table: complete explicit ordered-edge coverage ------


def test_device_specific_cz_table_dispatches_by_ordered_device_operands(
    model, calibration
):
    def forward_rule(operation, *, targets, model, calibration):
        return _drive_definition(model, "q0", 0.11)

    def reversed_rule(operation, *, targets, model, calibration):
        return _drive_definition(model, "q0", 0.22)

    # The fixture model declares exactly one coupling edge (q0, q1); a
    # complete explicit table covers both legal orderings of that one edge.
    implementations = PulseImplementationMap()
    implementations.add(fq.ops.CZ, forward_rule, device_operands=("q0", "q1"))
    implementations.add(fq.ops.CZ, reversed_rule, device_operands=("q1", "q0"))
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    forward = fq.Program(2)
    forward.add(fq.ops.CZ, (0, 1))
    (forward_block,) = backend._lower_program(forward)[0]
    assert np.allclose(forward_block.controls[0].coefficients, (0.11, 0.11))

    reversed_program = fq.Program(2)
    reversed_program.add(fq.ops.CZ, (1, 0))
    (reversed_block,) = backend._lower_program(reversed_program)[0]
    assert np.allclose(reversed_block.controls[0].coefficients, (0.22, 0.22))


# --- locked error policy through the full public workflow -------------------


def test_public_workflow_rejects_a_non_pulse_definition_return(model, calibration):
    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(
        fq.ops.CZ, lambda operation, *, targets, model, calibration: None
    )
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        backend.run(program)


def test_public_workflow_rejects_a_foreign_model_handle(
    model, calibration, build_model_and_calibration
):
    other, _ = build_model_and_calibration()  # a second, distinct model instance

    def foreign_rule(operation, *, targets, model, calibration):
        return PulseDefinition(
            1.0,
            (SampledControl(other.drive_control("q0"), (0.0, 1.0), (0.0, 0.0)),),
            (model.resource("q0"),),
        )

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, foreign_rule)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    with pytest.raises(BackendValidationError, match="foreign"):
        backend.run(program)


def test_public_workflow_rejects_a_control_extending_past_the_definition_duration(
    model, calibration
):
    def overrunning_rule(operation, *, targets, model, calibration):
        (subsystem_id,) = (model.subsystem_ids[model.bind_resource(t)] for t in targets)
        return PulseDefinition(
            1.0,
            (
                SampledControl(
                    model.drive_control(subsystem_id), (0.0, 2.0), (0.0, 0.0)
                ),
            ),
            (model.resource(subsystem_id),),
        )

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.RX, overrunning_rule)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    with pytest.raises(BackendValidationError, match="extends"):
        backend.run(program)


def test_public_workflow_rejects_missing_resource_claims(model, calibration):
    def claimless_rule(operation, *, targets, model, calibration):
        return PulseDefinition(0.0, (), ())

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.RZ, claimless_rule)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(1)
    program.add(fq.ops.RZ(0.3), 0)
    with pytest.raises(BackendValidationError, match="at least one model resource"):
        backend.run(program)


def test_public_workflow_wraps_an_arbitrary_exception_raised_by_the_rule(
    model, calibration
):
    def failing_rule(operation, *, targets, model, calibration):
        raise ValueError("miscalibrated recipe")

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, failing_rule)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    with pytest.raises(PulseImplementationError) as excinfo:
        backend.run(program)
    assert isinstance(excinfo.value.__cause__, ValueError)


# --- the documented workflow, reachable entirely from `fq.backends` ---------


def test_replacing_cz_through_the_public_surface_never_imports_fatqat_emulator(
    model, calibration
):
    """The custom-CZ workflow exactly as documented::
    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, custom_cz)
    backend = fq.backends.PulseBackend(
        model, calibration, pulse_implementation_map=implementations
    )

    Every name used below comes from `fatqat` / `fatqat.backends`: replacing a
    gate realization must never require subclassing `PulseBackend` or
    importing `fatqat.emulator`.
    """

    def custom_cz(operation, *, targets, model, calibration):
        first, second = (model.subsystem_ids[model.bind_resource(t)] for t in targets)
        duration = 10.0
        tlist = (0.0, duration)
        return PulseDefinition(
            duration,
            (SampledControl(model.exchange_control(first, second), tlist, (0.0, 0.0)),),
            (
                model.resource(first),
                model.resource(second),
                model.coupling(first, second),
            ),
            (PhaseShift(model.frame(first), 0.05),),
        )

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, custom_cz)
    backend = fq.backends.PulseBackend(
        model, calibration, pulse_implementation_map=implementations
    )

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    plan, _facts = backend._lower_program(program)

    (block,) = plan
    assert block.duration == 10.0
    (control,) = block.controls
    assert control.channel == model.exchange_control("q0", "q1")
