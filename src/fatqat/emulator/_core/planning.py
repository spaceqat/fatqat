"""Single-pass model-neutral preparation values and pulse lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..._backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_confusions,
    _resolve_condition,
)
from ..._backends.steps import MeasurementStep, ResetStep
from ..._index_allocation import _ClassicalAllocation, _EngineAllocation
from ...errors import BackendValidationError
from ...implementation._operation_registry import _select_implementation
from ...noise import NoiseModel
from ...noise.lindblad import (
    LindbladImplementationMap,
    resolve_lindblad_operators,
)
from ...operations.measurement import Measurement
from ...program import _AppliedOperation
from ...registers import RegisterRef
from ...resource_layout import ResourceLayout
from .engine import PulsePlanStep
from .lindblad import ResolvedLindbladTerm, bind_lindblad_operators
from .pulse import (
    PhaseShift,
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
    _invoke_pulse_rule,
)
from .target import (
    ResourceClaim,
    _ControlBinding,
    _PreparedControlBinding,
    _PulseTarget,
)

__all__ = [
    "PulsePlanFacts",
    "PulsePlanStep",
    "_PreparedPulseProgram",
]


@dataclass(frozen=True, slots=True)
class PulsePlanFacts:
    """Complete shared facts derived once from a finished pulse plan."""

    has_measurement: bool
    written_clbits: frozenset[int] = frozenset()
    has_reset: bool = False
    has_conditions: bool = False
    has_nonzero_evolution: bool = False
    has_potentially_active_lindblad: bool = False


@dataclass(frozen=True, slots=True)
class _PulseLoweringContext:
    """Run-local target and engine lookups derived during preparation."""

    resource_layout: ResourceLayout
    engine_allocation: _EngineAllocation
    classical_allocation: _ClassicalAllocation


@dataclass(frozen=True, slots=True)
class _PreparedPulseProgram:
    """One immutable run-lifetime product of the preparation template."""

    plan: tuple[PulsePlanStep, ...]
    facts: PulsePlanFacts
    resource_layout: ResourceLayout
    engine_allocation: _EngineAllocation
    classical_allocation: _ClassicalAllocation
    background_noise: tuple[ResolvedLindbladTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", tuple(self.plan))
        object.__setattr__(self, "background_noise", tuple(self.background_noise))


def _lower_measurement(
    step: Measurement,
    reported_digit_maps: tuple[tuple[int, ...], ...],
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    measured_indices, classical_indices, confusions = _lower_measurement_boundary(
        step,
        reported_digit_maps,
        resource_layout,
        engine_allocation,
        classical_allocation,
        noise_model,
    )
    return MeasurementStep(
        measured_indices=measured_indices,
        classical_indices=classical_indices,
        reported_digit_maps=reported_digit_maps,
        confusions=confusions,
    )


def _expectation_confusions(
    factor_refs: tuple[RegisterRef, ...],
    reported_digit_maps: tuple[tuple[int, ...], ...],
    resource_layout: ResourceLayout,
    noise_model: NoiseModel,
) -> tuple[Any, ...] | None:
    """Resolve readout confusion for a pulse expectation measurement."""
    return _resolve_confusions(
        factor_refs,
        reported_digit_maps,
        resource_layout,
        noise_model,
    )


def _build_expectation_measurement(
    factor_refs: tuple[RegisterRef, ...],
    measured_indices: tuple[int, ...],
    *,
    scratch_start: int,
    target: _PulseTarget,
    resource_layout: ResourceLayout,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Build one private terminal readout through normal pulse routing."""
    reported_digit_maps = tuple(
        target.reported_digit_map(resource_layout.device_label(ref))
        for ref in factor_refs
    )
    return MeasurementStep(
        measured_indices=measured_indices,
        classical_indices=tuple(
            range(scratch_start, scratch_start + len(measured_indices))
        ),
        reported_digit_maps=reported_digit_maps,
        confusions=_expectation_confusions(
            factor_refs,
            reported_digit_maps,
            resource_layout,
            noise_model,
        ),
    )


def _lower_reset(
    step: _AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
) -> ResetStep:
    return _lower_reset_boundary(
        step,
        resource_layout,
        engine_allocation,
        classical_allocation,
    )


def _bind_definition(
    target: _PulseTarget,
    definition: PulseDefinition,
    device_operands: tuple[object, ...],
) -> tuple[tuple[_ControlBinding, ...], tuple[ResourceClaim, ...]]:
    """Bind every gate-definition address exactly once."""
    occurrence = target.bind_gate_operands(device_operands)
    allowed_operands = set(occurrence.device_operands)
    allowed_claims = set(occurrence.claims)
    control_bindings = tuple(
        target.bind_control(control.channel) for control in definition.controls
    )
    frame_bindings = []
    for action in definition.post_actions:
        if isinstance(action, PhaseShift):
            frame_bindings.append(target.bind_frame(action.frame))
        else:
            frame_bindings.append(target.bind_frame(action.first))
            frame_bindings.append(target.bind_frame(action.second))
    if any(
        not set(binding.device_operands) <= allowed_operands
        for binding in frame_bindings
    ) or any(
        not set(binding.device_operands) <= allowed_operands
        or (
            not binding.allows_additional_claims
            and not set(binding.claims) <= allowed_claims
        )
        for binding in control_bindings
    ):
        raise BackendValidationError(
            "pulse definition addresses a subsystem outside its gate occurrence"
        )
    target.validate_pulse_controls(
        definition.controls,
        control_bindings,
        definition.duration,
    )
    claims = tuple(
        dict.fromkeys(
            claim
            for binding in (occurrence, *control_bindings, *frame_bindings)
            for claim in binding.claims
        )
    )
    return control_bindings, claims


def _prepare_control_bindings(
    bindings: tuple[_ControlBinding, ...],
    engine_allocation: _EngineAllocation,
) -> tuple[_PreparedControlBinding, ...]:
    """Translate target-physical controls to prepared numerical extents once."""
    return tuple(
        _PreparedControlBinding(
            binding.kind,
            tuple(
                engine_allocation.engine_index(operand)
                for operand in binding.device_operands
            ),
        )
        for binding in bindings
    )


def _resolve_operation_noise(
    step: _AppliedOperation,
    *,
    target: _PulseTarget,
    context: _PulseLoweringContext,
    noise_model: NoiseModel,
    implementation_map: LindbladImplementationMap,
) -> tuple[tuple[ResolvedLindbladTerm, ...], tuple[int, ...]]:
    terms: list[ResolvedLindbladTerm] = []
    engine_indices: list[int] = []
    for channel, extent in noise_model._noise_for_occurrence(
        type(step.operation),
        step.targets,
        context.resource_layout,
    ):
        if len(extent) != 1:
            raise BackendValidationError(
                f"{type(channel).__name__} selected a {len(extent)}-subsystem "
                "extent; pulse Lindblad noise must be local to one subsystem"
            )
        extent_indices = tuple(
            context.engine_allocation.engine_index(
                context.resource_layout.device_label(resource)
            )
            for resource in extent
        )
        engine_indices.extend(extent_indices)
        terms.extend(
            bind_lindblad_operators(
                resolve_lindblad_operators(
                    channel,
                    implementation_map=implementation_map,
                    physical_dimension=target.local_dimension,
                ),
                engine_indices=extent_indices,
            )
        )
    return tuple(terms), tuple(dict.fromkeys(engine_indices))


def _lower_gate(
    step: _AppliedOperation,
    *,
    target: _PulseTarget,
    context: _PulseLoweringContext,
    gate_implementation_map: PulseImplementationMap,
    noise_model: NoiseModel,
    lindblad_implementation_map: LindbladImplementationMap,
) -> PulseBlock:
    """Lower one ordinary operation using the shared target contracts."""
    device_operands = context.resource_layout.device_labels_for(step.targets)
    rule = _select_implementation(
        step.operation,
        device_operands,
        gate_implementation_map,
    )
    definition = _invoke_pulse_rule(
        rule,
        step.operation,
        device_operands=device_operands,
    )
    target_control_bindings, claims = _bind_definition(
        target,
        definition,
        device_operands,
    )
    noise, noise_indices = _resolve_operation_noise(
        step,
        target=target,
        context=context,
        noise_model=noise_model,
        implementation_map=lindblad_implementation_map,
    )
    target_indices = tuple(
        dict.fromkeys(
            (
                *(
                    context.engine_allocation.engine_index(
                        context.resource_layout.device_label(resource)
                    )
                    for resource in step.targets
                ),
                *noise_indices,
            )
        )
    )
    return PulseBlock(
        duration=definition.duration,
        controls=definition.controls,
        control_bindings=_prepare_control_bindings(
            target_control_bindings, context.engine_allocation
        ),
        resource_claims=claims,
        post_actions=definition.post_actions,
        condition=_resolve_condition(step.condition, context.classical_allocation),
        noise=noise,
        target_indices=target_indices,
    )


def _lower_direct(
    step: _AppliedOperation,
    *,
    target: _PulseTarget,
    context: _PulseLoweringContext,
) -> PulseBlock:
    """Bind one direct-control occurrence without backend-family plumbing."""
    controls = step.operation.controls
    target_bindings = tuple(
        target.bind_control(control.channel) for control in controls
    )
    target.validate_pulse_controls(controls, target_bindings, step.operation.duration)
    claims = tuple(
        dict.fromkeys(claim for binding in target_bindings for claim in binding.claims)
    )
    bindings = _prepare_control_bindings(target_bindings, context.engine_allocation)
    return PulseBlock(
        duration=step.operation.duration,
        controls=controls,
        control_bindings=bindings,
        resource_claims=claims,
        condition=_resolve_condition(step.condition, context.classical_allocation),
        target_indices=tuple(
            dict.fromkeys(
                engine_index
                for binding in bindings
                for engine_index in binding.engine_indices
            )
        ),
    )


def _resolve_background_noise(
    *,
    target: _PulseTarget,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    noise_model: NoiseModel,
    implementation_map: LindbladImplementationMap,
) -> tuple[ResolvedLindbladTerm, ...]:
    """Resolve background terms once in complete target-ordinal order."""
    terms: list[ResolvedLindbladTerm] = []
    for engine_index, device_label in enumerate(engine_allocation.device_operands):
        try:
            resource = resource_layout._ref_for_label(device_label)
        except KeyError:
            resource = None
        for channel in noise_model._background_noise_for(resource, device_label):
            terms.extend(
                bind_lindblad_operators(
                    resolve_lindblad_operators(
                        channel,
                        implementation_map=implementation_map,
                        physical_dimension=target.local_dimension,
                    ),
                    engine_indices=(engine_index,),
                )
            )
    return tuple(terms)


def _derive_plan_facts(
    plan: tuple[PulsePlanStep, ...],
    background_noise: tuple[ResolvedLindbladTerm, ...],
) -> PulsePlanFacts:
    nonzero_blocks = tuple(
        step for step in plan if isinstance(step, PulseBlock) and step.duration > 0.0
    )
    return PulsePlanFacts(
        has_measurement=any(isinstance(step, MeasurementStep) for step in plan),
        written_clbits=frozenset(
            classical_index
            for step in plan
            if isinstance(step, MeasurementStep)
            for classical_index in step.classical_indices
        ),
        has_reset=any(isinstance(step, ResetStep) for step in plan),
        has_conditions=any(
            getattr(step, "condition", None) is not None for step in plan
        ),
        has_nonzero_evolution=bool(nonzero_blocks),
        has_potentially_active_lindblad=(
            bool(background_noise) and bool(nonzero_blocks)
        )
        or any(bool(step.noise) for step in nonzero_blocks),
    )
