"""Unplaced pulse lowering and shared-boundary preservation."""

from copy import deepcopy

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator._core.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
)
from fatqat.emulator import SampledWaveform
from fatqat.emulator.superconducting import TransmonModel
from fatqat.emulator.superconducting.realization import (
    default_transmon_gate_implementation_map,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import NoiseModel


@pytest.fixture(name="make_backend")
def make_backend_fixture(model, calibration):
    del calibration

    def build(noise=None, gate_implementation_map=None):
        return TransmonEmulator(
            model,
            noise=noise,
            gate_implementation_map=gate_implementation_map,
        )

    return build


def test_lowering_produces_unplaced_blocks_and_preserves_boundaries_and_guards(
    backend,
):
    program = fq.Program(2, 1)
    program.add(ops.RX(0.4), 0)
    program.measure(0, 0)
    program.add(ops.RZ(0.2), 1, condition=(0, 0))
    program.add(ops.Reset, 1, condition=(0, 0))
    prepared = backend._prepare_program(program)
    plan, facts = prepared.plan, prepared.facts

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


def test_superconducting_backend_lowers_direct_drive_operation(backend):
    program = fq.Program(2)
    operation = ops.PulseOperation(
        1.0,
        (
            PulseControl(
                backend.model.control.drive("q0"),
                SampledWaveform((0.0, 1.0), (0.0, 0.1j)),
            ),
        ),
    )
    program.add(operation)

    (block,) = backend._prepare_program(program).plan
    assert block.controls == operation.controls
    assert block.control_bindings[0].engine_indices == (0,)
    assert block.target_indices == (0,)


def test_pulse_plan_facts_distinguish_virtual_and_elapsed_evolution(backend):
    virtual = fq.Program(1)
    virtual.add(ops.RZ(0.2), 0)
    facts = backend._prepare_program(virtual).facts
    assert facts.has_nonzero_evolution is False

    driven = fq.Program(1)
    driven.add(ops.RX(0.2), 0)
    facts = backend._prepare_program(driven).facts
    assert facts.has_nonzero_evolution is True


def test_edgeless_source_is_unsupported_while_reversed_cz_is_valid(
    backend, model_document, calibration_document
):
    disconnected_document = model_document
    disconnected_document["system"]["control_edges"] = []
    disconnected = TransmonModel.from_document(disconnected_document)
    disconnected_backend = TransmonEmulator(disconnected)
    iswap = fq.Program(2)
    iswap.add(ops.iSwap, (0, 1))
    with pytest.raises(UnsupportedOperationError, match="not supported"):
        disconnected_backend.run(iswap)

    reversed_cz = fq.Program(2)
    reversed_cz.add(ops.CZ, (1, 0))
    (block,) = backend._prepare_program(reversed_cz).plan
    assert block.controls[0].channel == backend.model.control.detuning("q1")


# --- shared measurement-lowering boundary: confusion validation parity -----


def test_pulse_measurement_confusion_must_match_the_reported_bit_dimension(
    make_backend,
):
    noise = NoiseModel()
    noise.add(fq.noise.ReadoutConfusion(np.eye(3)), targets="q0")
    backend = make_backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    # Routed through the shared boundary helper (backend_utils._resolve_confusions):
    # pulse's literal (0, 1, 1) reported-digit map implies reported dimension
    # 2, so a 3x3 confusion is rejected with the same "reported classical
    # dimension" message the matrix family raises for an analogous mismatch
    # (see the simulator readout-confusion dimension-mismatch coverage).
    with pytest.raises(BackendValidationError, match="reported classical dimension"):
        backend._prepare_program(program)


def test_pulse_measurement_accepts_a_correctly_shaped_confusion_matrix(
    make_backend,
):
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = NoiseModel()
    noise.add(fq.noise.ReadoutConfusion(always_flip), targets="q0")
    backend = make_backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    plan = backend._prepare_program(program).plan
    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    # Pulse always stores its literal qutrit-to-bit map, unlike the matrix
    # family's `None` identity default, whether or not confusion is present.
    assert measurement.reported_digit_maps == ((0, 1, 1),)
    assert np.array_equal(measurement.confusions[0], always_flip)


# --- lowering routed through PulseImplementationMap (custom CZ) ------------


def _constant_definition(model, subsystem_id):
    return PulseDefinition(
        1.0,
        (
            PulseControl(
                model.control.drive(subsystem_id),
                SampledWaveform((0.0, 1.0), (0.3, 0.3)),
            ),
        ),
    )


def test_a_copied_default_map_produces_the_same_plan_as_the_implicit_default(
    model, calibration
):
    explicit = TransmonEmulator(
        model,
        gate_implementation_map=default_transmon_gate_implementation_map(
            model=model, calibration=calibration
        ),
    )
    implicit = TransmonEmulator(model)
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    explicit_plan = explicit._prepare_program(program).plan
    implicit_plan = implicit._prepare_program(program).plan
    (explicit_block,) = explicit_plan
    (implicit_block,) = implicit_plan
    assert explicit_block.duration == implicit_block.duration
    assert tuple(
        (claim.kind, claim.ordinal) for claim in explicit_block.resource_claims
    ) == tuple((claim.kind, claim.ordinal) for claim in implicit_block.resource_claims)
    for a, b in zip(explicit_block.controls, implicit_block.controls):
        assert a.channel == b.channel
        assert np.allclose(a.waveform.values, b.waveform.values)


def test_custom_cz_rule_changes_emitted_controls_without_touching_program_or_adapter(
    model, calibration
):
    def custom_cz(operation, *, device_operands):
        del operation
        first_id, _second_id = device_operands
        return _constant_definition(model, first_id)

    implementations = default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    implementations.remove(ops.CZ)
    implementations.add(ops.CZ, custom_cz)
    backend = TransmonEmulator(model, gate_implementation_map=implementations)

    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    plan = backend._prepare_program(program).plan
    (block,) = plan

    assert block.duration == 1.0
    (control,) = block.controls
    assert control.channel == model.control.drive("q0")
    assert np.allclose(control.waveform.values, (0.3, 0.3))
    # target_indices comes from the occurrence's own program targets, not
    # from whatever the rule's returned definition happens to claim: the
    # custom rule above claims only "q0", yet both CZ operands still appear.
    assert block.target_indices == (0, 1)


def test_guarded_custom_rule_attaches_condition_only_to_the_block_not_the_definition(
    model, calibration
):
    shared_definition = _constant_definition(model, "q0")

    def reusable_rule(operation, *, device_operands):
        del operation
        assert device_operands == ("q0",)
        return shared_definition

    implementations = PulseImplementationMap()
    implementations.add(ops.RX, reusable_rule)
    backend = TransmonEmulator(model, gate_implementation_map=implementations)

    program = fq.Program(1, 1)
    program.add(ops.RX(0.1), 0)
    program.measure(0, 0)
    program.add(ops.RX(0.1), 0, condition=(0, 1))
    plan = backend._prepare_program(program).plan
    unguarded, _measurement, guarded = plan

    assert unguarded.condition is None
    assert guarded.condition == ((0, 1),)
    # The rule returned the very same PulseDefinition both times; each
    # resulting PulseBlock is an independent occurrence built from it.
    assert unguarded.duration == guarded.duration == shared_definition.duration
    assert unguarded.controls == guarded.controls == shared_definition.controls
    assert not hasattr(shared_definition, "condition")


def test_mutating_the_callers_map_after_construction_does_not_affect_the_backend(
    make_backend, model, calibration
):
    implementations = default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    backend = make_backend(gate_implementation_map=implementations)

    implementations.remove(ops.CZ)
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    plan = backend._prepare_program(program).plan
    assert len(plan) == 1  # still resolves; the backend's copy is unaffected


def test_coarse_compiled_map_transfers_unchanged_but_rebuild_redesigns_drag(
    model, calibration, model_document
):
    source_map = default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    finer_document = deepcopy(model_document)
    finer_document["parameters"]["subsystems"]["q0"]["anharmonicity"] = -0.4
    finer = TransmonModel.from_document(finer_document)
    source_backend = TransmonEmulator(model, gate_implementation_map=source_map)
    transferred = TransmonEmulator(finer, gate_implementation_map=source_map)
    rebuilt = TransmonEmulator(
        finer,
        gate_implementation_map=default_transmon_gate_implementation_map(
            model=finer, calibration=calibration
        ),
    )
    program = fq.Program(1)
    program.add(ops.RX(0.7), 0)
    source_block = source_backend._prepare_program(program).plan[0]
    transferred_block = transferred._prepare_program(program).plan[0]
    rebuilt_block = rebuilt._prepare_program(program).plan[0]
    assert np.array_equal(
        source_block.controls[0].waveform.values,
        transferred_block.controls[0].waveform.values,
    )
    assert not np.allclose(
        transferred_block.controls[0].waveform.values,
        rebuilt_block.controls[0].waveform.values,
    )


def test_unsupported_operation_from_map_selection_raises_out_of_run_not_as_a_failed_job(
    make_backend,
):
    backend = make_backend(gate_implementation_map=PulseImplementationMap())
    program = fq.Program(1)
    program.add(ops.RX(0.3), 0)

    with pytest.raises(
        UnsupportedOperationError, match="not supported by this backend"
    ):
        backend.run(program)


def test_unsupported_device_operands_from_map_selection_raises_out_of_run(
    make_backend,
):
    implementations = PulseImplementationMap()
    implementations.add(ops.CZ, lambda *a, **k: None, device_operands=("q5", "q6"))
    backend = make_backend(gate_implementation_map=implementations)
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    with pytest.raises(UnsupportedOperationError, match="device operands"):
        backend.run(program)
