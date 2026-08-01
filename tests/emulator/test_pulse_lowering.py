"""Unplaced pulse lowering and shared-boundary preservation."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import MeasurementStep, ResetStep
from fatqat.emulator.backend import PulseBackend
from fatqat.emulator.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
)
from fatqat.emulator.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.emulator.superconducting_realization import (
    default_superconducting_pulse_implementation_map,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import NoiseModel


@pytest.fixture(name="make_backend")
def make_backend_fixture(model, calibration):
    def build(noise=None, pulse_implementation_map=None):
        return PulseBackend(
            model,
            calibration,
            noise=noise,
            pulse_implementation_map=pulse_implementation_map,
        )

    return build


def test_lowering_produces_unplaced_blocks_and_preserves_boundaries_and_guards(
    backend,
):
    program = fq.Program(2, 1)
    program.add(fq.ops.RX(0.4), 0)
    program.measure(0, 0)
    program.add(fq.ops.RZ(0.2), 1, condition=(0, 0))
    program.add(fq.ops.Reset, 1, condition=(0, 0))
    plan, facts = backend._lower_program(program)

    assert [type(step) for step in plan] == [
        PulseBlock,
        MeasurementStep,
        PulseBlock,
        ResetStep,
    ]
    assert plan[0].start_time is None
    assert plan[1].reported_digit_maps == ((0, 1, 1),)
    assert plan[2].condition == ((0, 0),)
    assert plan[3].condition == ((0, 0),)
    assert facts.has_measurement


def test_lowering_rejects_absent_edges_and_reversed_cz_orientation(
    backend, model_document, calibration_document
):
    disconnected_document = model_document
    disconnected_document["parameters"]["couplings"] = []
    disconnected = load_physics_model(disconnected_document)
    calibration_document["recipes"]["cz"]["edges"] = []
    disconnected_backend = PulseBackend(
        disconnected, load_calibration_spec(calibration_document, disconnected)
    )
    iswap = fq.Program(2)
    iswap.add(fq.ops.iSwap, (0, 1))
    with pytest.raises(BackendValidationError, match="no declared coupling"):
        disconnected_backend.run(iswap)

    reversed_cz = fq.Program(2)
    reversed_cz.add(fq.ops.CZ, (1, 0))
    with pytest.raises(BackendValidationError, match="orientation"):
        backend.run(reversed_cz)


# --- shared measurement-lowering boundary: confusion validation parity -----


def test_pulse_measurement_confusion_must_match_the_reported_bit_dimension(
    make_backend,
):
    noise = NoiseModel()
    noise.add_readout_error(np.eye(3), target="q0")
    backend = make_backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    # Routed through the shared boundary helper (backend_utils._resolve_confusions):
    # pulse's literal (0, 1, 1) reported-digit map implies reported dimension
    # 2, so a 3x3 confusion is rejected with the same "reported classical
    # dimension" message the matrix family raises for an analogous mismatch
    # (see tests/backend/test_readout_error.py::test_dimension_mismatch_rejected_at_lowering).
    with pytest.raises(BackendValidationError, match="reported classical dimension"):
        backend._lower_program(program)


def test_pulse_measurement_accepts_a_correctly_shaped_confusion_matrix(
    make_backend,
):
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = NoiseModel()
    noise.add_readout_error(always_flip, target="q0")
    backend = make_backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    plan, _facts = backend._lower_program(program)
    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    # Pulse always stores its literal qutrit-to-bit map, unlike the matrix
    # family's `None` identity default, whether or not confusion is present.
    assert measurement.reported_digit_maps == ((0, 1, 1),)
    assert np.array_equal(measurement.confusions[0], always_flip)


# --- lowering routed through PulseImplementationMap (custom CZ) ------------


def _constant_definition(model, subsystem_id):
    return PulseDefinition(
        1.0,
        (SampledControl(model.drive_control(subsystem_id), (0.0, 1.0), (0.3, 0.3)),),
        (model.resource(subsystem_id),),
    )


def test_a_copied_default_map_produces_the_same_plan_as_the_implicit_default(
    model, calibration
):
    explicit = PulseBackend(
        model,
        calibration,
        pulse_implementation_map=default_superconducting_pulse_implementation_map(),
    )
    implicit = PulseBackend(model, calibration)
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))

    explicit_plan, _ = explicit._lower_program(program)
    implicit_plan, _ = implicit._lower_program(program)
    (explicit_block,) = explicit_plan
    (implicit_block,) = implicit_plan
    assert explicit_block.duration == implicit_block.duration
    assert explicit_block.resource_claims == implicit_block.resource_claims
    for a, b in zip(explicit_block.controls, implicit_block.controls):
        assert a.channel == b.channel
        assert np.allclose(a.coefficients, b.coefficients)


def test_custom_cz_rule_changes_emitted_controls_without_touching_program_or_adapter(
    model, calibration
):
    def custom_cz(operation, *, targets, model, calibration):
        first_id, _second_id = (
            model.subsystem_ids[model.bind_resource(t)] for t in targets
        )
        return _constant_definition(model, first_id)

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, custom_cz)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    plan, _facts = backend._lower_program(program)
    (block,) = plan

    assert block.duration == 1.0
    (control,) = block.controls
    assert control.channel == model.drive_control("q0")
    assert np.allclose(control.coefficients, (0.3, 0.3))
    # target_indices comes from the occurrence's own logical targets, not
    # from whatever the rule's returned definition happens to claim: the
    # custom rule above claims only "q0", yet both CZ operands still appear.
    assert block.target_indices == (0, 1)


def test_guarded_custom_rule_attaches_condition_only_to_the_block_not_the_definition(
    model, calibration
):
    shared_definition = _constant_definition(model, "q0")

    def reusable_rule(operation, *, targets, model, calibration):
        return shared_definition

    implementations = PulseImplementationMap()
    implementations.add(fq.ops.RX, reusable_rule)
    backend = PulseBackend(model, calibration, pulse_implementation_map=implementations)

    program = fq.Program(1, 1)
    program.add(fq.ops.RX(0.1), 0)
    program.measure(0, 0)
    program.add(fq.ops.RX(0.1), 0, condition=(0, 1))
    plan, _facts = backend._lower_program(program)
    unguarded, _measurement, guarded = plan

    assert unguarded.condition is None
    assert guarded.condition == ((0, 1),)
    # The rule returned the very same PulseDefinition both times; each
    # resulting PulseBlock is an independent occurrence built from it.
    assert unguarded.duration == guarded.duration == shared_definition.duration
    assert unguarded.controls == guarded.controls == shared_definition.controls
    assert not hasattr(shared_definition, "condition")


def test_mutating_the_callers_map_after_construction_does_not_affect_the_backend(
    make_backend,
):
    implementations = default_superconducting_pulse_implementation_map()
    backend = make_backend(pulse_implementation_map=implementations)

    implementations.remove(fq.ops.CZ)
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    plan, _facts = backend._lower_program(program)
    assert len(plan) == 1  # still resolves; the backend's copy is unaffected


def test_unsupported_operation_from_map_selection_raises_out_of_run_not_as_a_failed_job(
    make_backend,
):
    backend = make_backend(pulse_implementation_map=PulseImplementationMap())
    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)

    with pytest.raises(
        UnsupportedOperationError, match="not supported by this backend"
    ):
        backend.run(program)


def test_unsupported_device_operands_from_map_selection_raises_out_of_run(
    make_backend,
):
    implementations = PulseImplementationMap()
    implementations.add(fq.ops.CZ, lambda *a, **k: None, device_operands=("q5", "q6"))
    backend = make_backend(pulse_implementation_map=implementations)
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))

    with pytest.raises(UnsupportedOperationError, match="device operands"):
        backend.run(program)
