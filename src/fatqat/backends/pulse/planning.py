"""Unplaced pulse-plan lowering for the SC pulse backend."""

from __future__ import annotations

from dataclasses import dataclass

from ..._engine_index_allocation import _EngineIndexAllocation
from ...backends.backend_utils import _resolve_condition
from ...backends.simulator_backend import _break_grouped_operations
from ...backends.steps import MeasurementStep, ResetStep
from ...errors import BackendValidationError
from ...noise import NoiseModel
from ...operations.barrier import BarrierGate
from ...operations.measurement import Measurement
from ...operations.reset import ResetGate
from ...program import AppliedOperation, Program
from ...resource_layout import ResourceLayout
from .resolved import PulseBlock, realize_native_operation
from .superconducting import CalibrationSpec, PhysicsModel

PulsePlanStep = PulseBlock | MeasurementStep | ResetStep


@dataclass(frozen=True)
class PulsePlanFacts:
    """Lowered-program facts needed for default pulse result requests."""

    has_measurement: bool
    has_reset: bool
    has_guarded_pulse: bool


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
        has_reset=any(isinstance(step, ResetStep) for step in plan),
        has_guarded_pulse=any(
            isinstance(step, PulseBlock) and step.condition is not None for step in plan
        ),
    )


def _lower_measurement(
    step: Measurement,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Preserve the shared measurement boundary in the unplaced pulse plan."""
    measured_indices = tuple(
        engine_index_allocation.subsystem_index(target) for target in step.targets
    )
    confusions = []
    for target in step.targets:
        confusion = noise_model.readout_error_for(target, resource_layout)
        if confusion is not None and confusion.shape != (2, 2):
            raise BackendValidationError(
                "pulse v0.1 readout confusion matrices must use reported bit dimension 2"
            )
        confusions.append(confusion)
    return MeasurementStep(
        measured_indices=measured_indices,
        classical_indices=tuple(
            engine_index_allocation.clbit_index(output) for output in step.outputs
        ),
        # The future qutrit engine collapses physical 0/1/2 outcomes, while
        # this shared boundary writes one reported bit before confusion.
        reported_digit_maps=tuple((0, 1, 1) for _ in measured_indices),
        confusions=(
            None
            if all(confusion is None for confusion in confusions)
            else tuple(confusions)
        ),
    )
