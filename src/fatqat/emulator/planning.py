"""Unplaced pulse-plan lowering for the SC pulse backend."""

from __future__ import annotations

from dataclasses import dataclass

from .._engine_index_allocation import _EngineIndexAllocation
from ..backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_condition,
)
from ..backends.steps import MeasurementStep, ResetStep
from ..noise import NoiseModel
from ..operations.measurement import Measurement
from ..program import AppliedOperation
from ..resource_layout import ResourceLayout
from .resolved import PulseBlock, realize_native_operation
from .superconducting import CalibrationSpec, PhysicsModel

PulsePlanStep = PulseBlock | MeasurementStep | ResetStep


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
    return _lower_reset_boundary(step, engine_index_allocation)


def _lower_gate(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    model: PhysicsModel,
    calibration: CalibrationSpec,
) -> PulseBlock:
    targets = tuple(
        model.resource(resource_layout.device_label(target)) for target in step.targets
    )
    return realize_native_operation(
        step.operation,
        targets,
        model=model,
        calibration=calibration,
        condition=_resolve_condition(step.condition, engine_index_allocation),
    )
