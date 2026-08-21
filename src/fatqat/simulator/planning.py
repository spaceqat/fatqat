"""Private lowering helpers for the matrix-family simulator backend."""

from __future__ import annotations

from math import prod
from typing import cast

from .._index_allocation import _ClassicalAllocation, _EngineAllocation
from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
)
from ..implementation import MatrixImplementationMap
from ..implementation._operation_registry import _select_implementation
from ..noise import ChannelImplementationMap, NoiseModel
from ..noise.base import _validate_kraus_shapes
from ..operations import Measurement, PulseOperation
from ..noise.loss import Loss
from ..program import AppliedOperation
from ..resource_layout import ResourceLayout
from .._backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_condition,
)
from .._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    ResetStep,
    PutStep,
    ResolvedStep,
)


def _lower_measurement(
    step: Measurement,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Lower one ``Measurement`` instruction into a ``MeasurementStep``."""
    # Only used to build reported_digit_maps below; _lower_measurement_boundary
    # independently resolves the authoritative measured_indices from step.targets.
    digit_map_indices = tuple(
        engine_allocation.engine_index(resource_layout.device_label(q))
        for q in step.targets
    )
    reported_digit_maps = tuple(
        tuple(range(engine_allocation.system_dims[measured]))
        for measured in digit_map_indices
    )
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
        confusions=confusions,
        # The identity map is the compatibility default (see steps.py); only
        # carry it explicitly when a confusion matrix needs it for the
        # reported-dimension check, so an identity, noise-free measurement
        # keeps the None default the numba fast path recognizes.
        reported_digit_maps=reported_digit_maps if confusions is not None else None,
    )


def _lower_reset(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
) -> ResetStep:
    """Lower one ``Reset`` operation after admission rejected attached noise."""
    return _lower_reset_boundary(
        step,
        resource_layout,
        engine_allocation,
        classical_allocation,
    )


def _lower_put(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    noise_model: NoiseModel,
) -> list[ResolvedStep]:
    """Lower one ``Put`` into a ``PutStep`` plus attached carrier loss.

    ``Put`` loads a fresh ``|0>`` atom into each currently-empty target site,
    reusing the engine's per-shot ``PutStep``. Attached ``Loss`` is
    emitted after the fill (loading inefficiency, S-C1): an atom that arrives
    and is immediately lost gives ``p_success = 1 - p``. Only ``Loss`` may
    attach to ``Put`` -- a Kraus channel on a reload has no meaning here.
    """
    condition = _resolve_condition(step.condition, classical_allocation)
    engine_indices = tuple(
        engine_allocation.engine_index(resource_layout.device_label(t))
        for t in step.targets
    )
    steps: list[ResolvedStep] = [
        PutStep(target_indices=engine_indices, condition=condition)
    ]
    for declaration, extent in noise_model._noise_for_occurrence(
        type(step.operation), step.targets, resource_layout
    ):
        loss = cast(Loss, declaration)
        extent_indices = tuple(
            engine_allocation.engine_index(resource_layout.device_label(target))
            for target in extent
        )
        steps.append(
            LossStep(
                target_indices=extent_indices,
                p=loss.p,
                condition=condition,
            )
        )
    return steps


def _lower_channels(
    operation_type,
    targets,
    condition,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    noise_model: NoiseModel,
    channel_map: ChannelImplementationMap,
) -> list[ResolvedStep]:
    """Lower the channel noise attached to one occurrence into steps.

    Shared by gate and Pair/Unpair lowering. Selection matches against the
    occurrence's program targets and/or resource-layout device operands
    (never engine indices); engine indices are used only for the emitted
    steps. A Loss becomes a per-carrier LossStep; any other (Kraus)
    channel becomes an ApplyChannelStep.
    """
    steps: list[ResolvedStep] = []
    for channel, extent in noise_model._noise_for_occurrence(
        operation_type, targets, resource_layout
    ):
        extent_indices = tuple(
            engine_allocation.engine_index(resource_layout.device_label(target))
            for target in extent
        )
        if isinstance(channel, Loss):
            steps.append(
                LossStep(
                    target_indices=extent_indices,
                    p=channel.p,
                    condition=condition,
                )
            )
            continue
        channel_rule = channel_map.get(type(channel))
        if channel_rule is None:
            raise UnsupportedOperationError(
                f"{type(channel).__name__} has no channel "
                "implementation on this backend"
            )
        kraus_ops = tuple(channel_rule(channel, targets=extent))
        extent_dim = prod(
            engine_allocation.system_dims[index] for index in extent_indices
        )
        _validate_kraus_shapes(kraus_ops, extent_dim, type(channel).__name__)
        steps.append(
            ApplyChannelStep(
                kraus_ops=kraus_ops,
                target_indices=extent_indices,
                condition=condition,
            )
        )
    return steps


def _lower_gate(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    impl_map: MatrixImplementationMap,
    noise_model: NoiseModel,
    channel_map: ChannelImplementationMap,
) -> list[ResolvedStep]:
    """Lower one ordinary-gate operation and its attached channel noise."""
    if isinstance(step.operation, PulseOperation):
        raise UnsupportedOperationError(
            "PulseOperation is not supported by the matrix simulator"
        )
    device_operands = resource_layout.device_labels_for(step.targets)
    engine_indices = tuple(
        engine_allocation.engine_index(resource_layout.device_label(t))
        for t in step.targets
    )
    condition = _resolve_condition(step.condition, classical_allocation)

    rule = _select_implementation(step.operation, device_operands, impl_map)
    try:
        matrix = rule(step.operation, targets=step.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(step.operation).__name__} raised: {exc}"
        ) from exc

    target_dims = tuple(engine_allocation.system_dims[i] for i in engine_indices)
    expected = prod(target_dims)
    if matrix.shape != (expected, expected):
        raise BackendValidationError(
            f"{type(step.operation).__name__} resolved to a "
            f"{matrix.shape} matrix, incompatible with target "
            f"dimensions {target_dims} (expected "
            f"{(expected, expected)})"
        )

    steps: list[ResolvedStep] = [
        ApplyMatrixStep(
            matrix=matrix,
            target_indices=engine_indices,
            condition=condition,
            kernel_key=rule._kernel_key(step.operation, targets=step.targets),
        )
    ]
    steps.extend(
        _lower_channels(
            type(step.operation),
            step.targets,
            condition,
            resource_layout,
            engine_allocation,
            noise_model,
            channel_map,
        )
    )
    return steps
