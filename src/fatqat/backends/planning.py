"""Private lowering helpers for the matrix-family simulator backend."""

from __future__ import annotations

from math import prod

from .._engine_index_allocation import _EngineIndexAllocation
from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
)
from ..implementation import DeviceOperands, ImplementationMap, MatrixImplementation
from ..noise import ChannelImplementationMap, NoiseModel
from ..noise.base import _validate_kraus_shapes
from ..operations import Measurement, Operation, ResetGate
from ..program import AppliedOperation
from ..resource_layout import ResourceLayout
from .backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_condition,
)
from .steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    MeasurementStep,
    ResetStep,
    ResolvedStep,
)


def _gate_implementation_for(
    operation: Operation, device_operands: DeviceOperands, impl_map: ImplementationMap
) -> MatrixImplementation:
    """Resolve the matrix rule for a gate operation on a device target key."""
    if not impl_map.supports(operation):
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported by this backend"
        )
    rule = impl_map.implementation_for(operation, device_operands=device_operands)
    if rule is None:
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported on device operands {device_operands}"
        )
    return rule


def _lower_measurement(
    step: Measurement,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Lower one ``Measurement`` instruction into a ``MeasurementStep``."""
    # Only used to build reported_digit_maps below; _lower_measurement_boundary
    # independently resolves the authoritative measured_indices from step.targets.
    digit_map_indices = tuple(
        engine_index_allocation.subsystem_index(q) for q in step.targets
    )
    reported_digit_maps = tuple(
        tuple(range(engine_index_allocation.system_dims[measured]))
        for measured in digit_map_indices
    )
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
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> ResetStep:
    """Lower one ``Reset`` operation, rejecting unimplemented channel noise."""
    if noise_model.channels_for(ResetGate, step.targets, resource_layout):
        raise UnsupportedOperationError(
            "channel noise attached to Reset is not supported yet"
        )
    return _lower_reset_boundary(step, engine_index_allocation)


def _lower_gate(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    impl_map: ImplementationMap,
    noise_model: NoiseModel,
    channel_map: ChannelImplementationMap,
) -> list[ResolvedStep]:
    """Lower one ordinary-gate operation and its attached channel noise."""
    device_operands = resource_layout.device_operands(step.targets)
    engine_indices = tuple(
        engine_index_allocation.subsystem_index(t) for t in step.targets
    )
    condition = _resolve_condition(step.condition, engine_index_allocation)

    rule = _gate_implementation_for(step.operation, device_operands, impl_map)
    try:
        matrix = rule(step.operation, targets=step.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(step.operation).__name__} raised: {exc}"
        ) from exc

    target_dims = tuple(engine_index_allocation.system_dims[i] for i in engine_indices)
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
            # Identity, not mechanics: the backend forwards which
            # implementation was selected; the engine alone decides
            # what (if anything) that means for kernel choice.
            kernel_key=rule._kernel_key(step.operation, targets=step.targets),
        )
    ]

    # Noise selection matches against the occurrence's logical targets
    # and/or resource-layout device operands (never engine indices);
    # engine indices are used only for the emitted ApplyChannelStep.
    for channel, extent in noise_model.channels_for(
        type(step.operation), step.targets, resource_layout
    ):
        channel_rule = channel_map.get(type(channel))
        if channel_rule is None:
            raise UnsupportedOperationError(
                f"{type(channel).__name__} has no channel "
                "implementation on this backend"
            )
        extent_indices = tuple(
            engine_index_allocation.subsystem_index(target) for target in extent
        )
        kraus_ops = tuple(channel_rule(channel, targets=extent))
        extent_dim = prod(
            engine_index_allocation.system_dims[index] for index in extent_indices
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
