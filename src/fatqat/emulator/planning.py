"""Unplaced pulse-plan lowering for the SC pulse backend."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .._engine_index_allocation import _EngineIndexAllocation
from ..backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_condition,
)
from ..backends.steps import MeasurementStep, ResetStep
from ..implementation._operation_registry import _select_implementation
from ..noise import NoiseModel
from ..noise import LindbladImplementationMap
from ..noise.lindblad import resolve_lindblad_operators
from ..operations.measurement import Measurement
from ..program import AppliedOperation
from ..resource_layout import ResourceLayout
from .engine import PulsePlanStep
from .lindblad import ResolvedLindbladTerm, bind_lindblad_operators
from .pulse import PulseBlock, PulseImplementationMap, _invoke_pulse_rule
from .superconducting import CalibrationSpec, SCTransmonModel

__all__ = ["PulsePlanFacts", "PulsePlanStep"]


@dataclass(frozen=True)
class PulsePlanFacts:
    """Lowered-program facts needed for default pulse result requests."""

    has_measurement: bool


def _lower_measurement(
    step: Measurement,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Preserve the shared measurement boundary in the unplaced pulse plan.

    The future qutrit engine collapses physical 0/1/2 outcomes, while this
    shared boundary writes one reported bit before confusion, so the
    reported-digit map is always the literal ``(0, 1, 1)`` qutrit-to-bit map
    - never the matrix family's ``None`` identity default. A confusion
    matrix selected for a measured subsystem must therefore be shaped
    ``(2, 2)``; that requirement is enforced by the shared boundary helper's
    general "shape must match the reported dimension" check, since
    ``max((0, 1, 1)) + 1 == 2`` is exactly pulse's v0.1 restriction.
    """
    reported_digit_maps = tuple((0, 1, 1) for _ in step.targets)
    measured_indices, classical_indices, confusions = _lower_measurement_boundary(
        step,
        reported_digit_maps,
        resource_layout,
        engine_index_allocation,
        noise_model,
    )
    return MeasurementStep(
        measured_indices=measured_indices,
        classical_indices=classical_indices,
        reported_digit_maps=reported_digit_maps,
        confusions=confusions,
    )


def _lower_reset(
    step: AppliedOperation,
    engine_index_allocation: _EngineIndexAllocation,
) -> ResetStep:
    """Lower reset to the engine-index boundary shared with matrix backends."""
    return _lower_reset_boundary(step, engine_index_allocation)


def _lower_gate(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    model: SCTransmonModel,
    calibration: CalibrationSpec,
    pulse_implementation_map: PulseImplementationMap,
    noise_model: NoiseModel,
    lindblad_implementation_map: LindbladImplementationMap,
) -> PulseBlock:
    """Lower one applied operation to a model-owned pulse occurrence.

    Device operands select a pulse implementation rule. The reusable
    definition returned by that rule is then enriched with this occurrence's
    lowered condition, resolved Lindblad noise, and engine target indices.
    Scheduling remains a later engine concern.
    """
    device_operands = resource_layout.device_labels_for(step.targets)
    rule = _select_implementation(
        step.operation, device_operands, pulse_implementation_map
    )
    targets = tuple(
        model.resource(device_operand) for device_operand in device_operands
    )
    definition = _invoke_pulse_rule(
        rule, step.operation, targets=targets, model=model, calibration=calibration
    )
    block = PulseBlock(
        model=model,
        duration=definition.duration,
        controls=definition.controls,
        resource_claims=definition.resource_claims,
        post_actions=definition.post_actions,
        condition=_resolve_condition(step.condition, engine_index_allocation),
    )
    noise_bindings, noise_target_indices = _lower_gate_noise(
        step,
        resource_layout,
        engine_index_allocation,
        model,
        noise_model,
        lindblad_implementation_map,
        block.duration,
    )
    target_indices = tuple(
        dict.fromkeys(
            (
                *(engine_index_allocation.subsystem_index(ref) for ref in step.targets),
                *noise_target_indices,
            )
        )
    )
    return dataclasses.replace(
        block, noise=noise_bindings, target_indices=target_indices
    )


def _lower_gate_noise(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    model: SCTransmonModel,
    noise_model: NoiseModel,
    lindblad_implementation_map: LindbladImplementationMap,
    duration: float,
) -> tuple[tuple[ResolvedLindbladTerm, ...], tuple[int, ...]]:
    """Resolve one gate occurrence's attached channels into engine-facing bindings.

    Noise selection matches against the occurrence's logical targets and/or
    resource-layout device operands (never engine indices), exactly like the
    matrix family's lowering; engine indices are used only for the emitted
    binding. Each channel resolves to backend-neutral Lindblad terms at the
    lowering boundary, using the realized block's own duration; concrete
    adapters receive no source descriptor, duration, or probability.
    """
    bindings: list[ResolvedLindbladTerm] = []
    target_indices: list[int] = []
    for channel, extent in noise_model.channels_for(
        type(step.operation), step.targets, resource_layout
    ):
        target_indices.extend(
            engine_index_allocation.subsystem_index(ref) for ref in extent
        )
        model_indices = tuple(
            model.bind_resource(model.resource(device_operand))
            for device_operand in resource_layout.device_labels_for(extent)
        )
        bindings.extend(
            bind_lindblad_operators(
                resolve_lindblad_operators(
                    channel,
                    implementation_map=lindblad_implementation_map,
                    physical_dimension=model.physical_dimension,
                    duration=duration,
                ),
                model_ordinals=model_indices,
            )
        )
    return tuple(bindings), tuple(dict.fromkeys(target_indices))
