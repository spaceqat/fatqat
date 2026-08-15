"""Shared direct and ordinary pulse lowering contracts."""

import inspect

import pytest

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core import planning
from fatqat.emulator._core.pulse import PulseDefinition, PulseImplementationMap
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator._core.target import (
    _ControlAddress,
    _ControlBinding,
    _TargetClaim,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.waveforms import SampledWaveform

from tests.emulator._core.test_backend_template import (
    _CountingTarget,
    _TemplateBackend,
    _gate_map,
)


def _control(target, label, *, duration=0.5):
    return PulseControl(
        target.model.drive(label),
        SampledWaveform((0.0, duration), (0.0, 0.2)),
    )


def _direct(target, label, *, duration=0.5):
    return fq.ops.PulseOperation(
        duration,
        (_control(target, label, duration=duration),),
    )


def test_direct_lowering_binds_each_control_once_and_stores_exact_result():
    target = _CountingTarget()
    backend = _TemplateBackend(target)
    program = fq.Program(2)
    operation = _direct(target, "q1")
    program.add(operation)

    prepared = backend._prepare_program(program)
    block = prepared.plan[0]
    assert target.bind_control_calls == 1
    assert block.control_bindings[0].engine_indices == (1,)
    assert block.controls == operation.controls
    assert block.target_indices == (1,)
    assert block.resource_claims == target.bound_controls[0].claims


def test_empty_gate_map_rejects_ordinary_gate_without_a_none_sentinel():
    backend = _TemplateBackend(_CountingTarget())
    program = fq.Program(1)
    program.add(fq.ops.X, 0)
    with pytest.raises(UnsupportedOperationError, match="XGate"):
        backend._prepare_program(program)


def test_custom_gate_binds_occurrence_control_condition_and_targets_once():
    target = _CountingTarget()
    backend = _TemplateBackend(target, gate_map=_gate_map(target))
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0, condition=(program.classical_registers[0][0], 1))

    prepared = backend._prepare_program(program)
    block = prepared.plan[0]
    assert target.bind_control_calls == 1
    assert target.validate_control_calls == 1
    assert block.condition == ((0, 1),)
    assert block.target_indices == (0,)
    assert block.control_bindings[0].engine_indices == (0,)
    assert block.resource_claims == target.bound_controls[0].claims


def test_gate_occurrence_control_and_frame_each_bind_once():
    target = _CountingTarget()
    backend = _TemplateBackend(
        target,
        gate_map=_gate_map(target, with_frame=True),
    )
    program = fq.Program(1)
    program.add(fq.ops.X, 0)

    backend._prepare_program(program)

    assert target.bind_gate_operands_calls == 1
    assert target.bind_control_calls == 1
    assert target.bind_frame_calls == 1


def test_gate_definition_cannot_escape_its_occurrence():
    target = _CountingTarget()
    implementation_map = PulseImplementationMap()

    def escape(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.5,
            (_control(target, "q1"),),
        )

    implementation_map.add(fq.ops.X, escape)
    backend = _TemplateBackend(target, gate_map=implementation_map)
    program = fq.Program(2)
    program.add(fq.ops.X, 0)
    with pytest.raises(BackendValidationError, match="outside its gate occurrence"):
        backend._prepare_program(program)


class _GlobalControlTarget(_CountingTarget):
    def bind_control(self, reference):
        if reference.kind != "global":
            return super().bind_control(reference)
        self.bind_control_calls += 1
        return _ControlBinding(
            "global",
            self.device_labels,
            self._claims,
        )


class _PairControlTarget(_CountingTarget):
    def bind_control(self, reference):
        if reference.kind != "pair":
            return super().bind_control(reference)
        self.bind_control_calls += 1
        ordinals = tuple(
            self.device_labels.index(operand) for operand in reference.operands
        )
        binding = _ControlBinding(
            "pair",
            tuple(reference.operands),
            tuple(self._claims[ordinal] for ordinal in ordinals),
        )
        self.bound_controls.append(binding)
        return binding


def test_direct_multi_target_binding_translates_every_target_ordinal_once():
    target = _PairControlTarget()
    control = PulseControl(
        _ControlAddress("fake", "pair", ("q0", "q1")),
        SampledWaveform((0.0, 0.5), (0.0, 0.2)),
    )
    program = fq.Program(2)
    program.add(fq.ops.PulseOperation(0.5, (control,)))

    block = _TemplateBackend(target)._prepare_program(program).plan[0]

    assert target.bind_control_calls == 1
    assert block.control_bindings[0].engine_indices == (0, 1)
    assert block.target_indices == (0, 1)


def test_global_control_cannot_escape_a_narrow_gate_occurrence():
    target = _GlobalControlTarget()
    implementation_map = PulseImplementationMap()

    def global_drive(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.5,
            (
                PulseControl(
                    _ControlAddress("fake", "global", ()),
                    SampledWaveform((0.0, 0.5), (0.0, 0.2)),
                ),
            ),
        )

    implementation_map.add(fq.ops.X, global_drive)
    backend = _TemplateBackend(target, gate_map=implementation_map)
    program = fq.Program(2)
    program.add(fq.ops.X, 0)

    with pytest.raises(BackendValidationError, match="outside its gate occurrence"):
        backend._prepare_program(program)


def test_global_control_can_cover_its_whole_gate_occurrence():
    target = _GlobalControlTarget(("q0", "q1"))
    implementation_map = PulseImplementationMap()

    def global_drive(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.5,
            (
                PulseControl(
                    _ControlAddress("fake", "global", ()),
                    SampledWaveform((0.0, 0.5), (0.0, 0.2)),
                ),
            ),
        )

    implementation_map.add(fq.ops.CZ, global_drive)
    backend = _TemplateBackend(target, gate_map=implementation_map)
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))

    block = backend._prepare_program(program).plan[0]

    assert block.control_bindings[0].engine_indices == (0, 1)
    assert block.target_indices == (0, 1)


class _SharedActuatorTarget(_CountingTarget):
    def __init__(self):
        super().__init__(("q0", "q1"))
        self._actuator = _TargetClaim(object(), "actuator", 0)

    def bind_control(self, reference):
        binding = super().bind_control(reference)
        return _ControlBinding(
            binding.kind,
            binding.device_operands,
            (self._actuator,),
        )

    def validate_pulse_controls(self, controls, bindings, block_duration):
        super().validate_pulse_controls(controls, bindings, block_duration)
        if len({binding.device_operands for binding in bindings}) > 1:
            raise BackendValidationError(
                "shared actuator accepts one target per concurrent block"
            )


def test_target_validator_rejects_incompatible_concurrent_controls_once():
    target = _SharedActuatorTarget()
    backend = _TemplateBackend(target)
    operation = fq.ops.PulseOperation(
        0.5,
        (_control(target, "q0"), _control(target, "q1")),
    )
    program = fq.Program(2)
    program.add(operation)
    with pytest.raises(BackendValidationError, match="shared actuator"):
        backend._prepare_program(program)
    assert target.bind_control_calls == 2
    assert target.validate_control_calls == 1


def test_shared_claim_serializes_separate_direct_blocks():
    target = _SharedActuatorTarget()
    backend = _TemplateBackend(target)
    program = fq.Program(2)
    program.add(_direct(target, "q0"))
    program.add(_direct(target, "q1"))
    prepared = backend._prepare_program(program)
    run = schedule_pulse_run(prepared.plan, boundary_time=0.0)
    assert run.starts == (0.0, 0.5)


def test_barrier_reset_measurement_and_fact_derivation_share_one_pass():
    target = _CountingTarget()
    backend = _TemplateBackend(target)
    program = fq.Program(1, 1)
    program.add(fq.ops.Barrier, 0)
    program.add(fq.ops.Reset, 0, condition=(program.classical_registers[0][0], 1))
    program.measure(0, 0)

    prepared = backend._prepare_program(program)
    assert len(prepared.plan) == 2
    assert prepared.facts.has_reset
    assert prepared.facts.has_measurement
    assert prepared.facts.has_conditions
    assert not prepared.facts.has_nonzero_evolution
    assert prepared.plan[-1].reported_digit_maps == ((0, 1),)


def test_shared_lowering_has_no_family_context_or_direct_hook_seams():
    backend_source = inspect.getsource(_TemplateBackend.__mro__[1])
    planning_source = inspect.getsource(planning)
    for removed in (
        "_LoweringContext",
        "_lower_direct_operation",
        "_direct_control_target_indices",
        "_lowering_model",
        "_lower_gate_noise",
    ):
        assert removed not in backend_source
    assert "_LoweringContext" not in planning_source
