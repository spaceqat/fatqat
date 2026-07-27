"""Unplaced pulse-plan lowering for the SC pulse backend."""

from __future__ import annotations

from dataclasses import dataclass

from .._engine_index_allocation import _EngineIndexAllocation
from ..backends.backend_utils import _lower_measurement_boundary, _resolve_condition
from ..backends.steps import MeasurementStep, ResetStep
from ..noise import NoiseModel
from ..operations.barrier import BarrierGate
from ..operations.measurement import Measurement
from ..operations.reset import ResetGate
from ..program import AppliedOperation, Program
from ..resource_layout import ResourceLayout
from .resolved import PulseBlock, realize_native_operation
from .superconducting import CalibrationSpec, PhysicsModel
from ..backends.view_normalization import _break_grouped_operations

PulsePlanStep = PulseBlock | MeasurementStep | ResetStep


@dataclass(frozen=True)
class PulsePlanFacts:
    """Lowered-program facts needed for default pulse result requests."""

    has_measurement: bool


def lower_program(
    program: Program,
    *,
    model: PhysicsModel,
    calibration: CalibrationSpec,
    noise_model: NoiseModel,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
) -> tuple[list[PulsePlanStep], PulsePlanFacts]:
    """Lower scalar program operations into an ordered, entirely unplaced plan."""
    plan: list[PulsePlanStep] = []
    for step in _break_grouped_operations(program.operations):
        if isinstance(step, Measurement):
            plan.append(
                _lower_measurement(
                    step, resource_layout, engine_index_allocation, noise_model
                )
            )
        elif isinstance(step, AppliedOperation):
            if isinstance(step.operation, BarrierGate):
                continue
            if isinstance(step.operation, ResetGate):
                plan.append(
                    ResetStep(
                        reset_indices=tuple(
                            engine_index_allocation.subsystem_index(target)
                            for target in step.targets
                        ),
                        condition=_resolve_condition(
                            step.condition, engine_index_allocation
                        ),
                    )
                )
                continue
            targets = tuple(
                model.resource(resource_layout.device_label(target))
                for target in step.targets
            )
            plan.append(
                realize_native_operation(
                    step.operation,
                    targets,
                    model=model,
                    calibration=calibration,
                    condition=_resolve_condition(
                        step.condition, engine_index_allocation
                    ),
                )
            )
    return plan, PulsePlanFacts(
        has_measurement=any(isinstance(step, MeasurementStep) for step in plan),
    )


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
