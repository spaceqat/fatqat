"""Private lowering helpers for the matrix-family simulator backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from math import prod
from typing import Any, cast

import numpy as np

from .._index_allocation import _ClassicalAllocation, _EngineAllocation
from .._parameter_binding import _discover_parameters
from ..errors import UnsupportedOperationError
from ..implementation import MatrixImplementationMap
from ..implementation.matrix_map import _build_matrix
from ..implementation.base_map import _select_implementation
from ..implementation.matrices import _H, _SDG
from ..noise import NoiseModel
from ..noise.base import _validate_kraus_shapes
from .. import operations as ops
from ..operations import Measurement, PulseOperation
from ..noise.loss import Loss
from ..parameters import Parameter
from ..program import _AppliedOperation
from ..registers import RegisterRef
from ..resource_layout import ResourceLayout
from .._backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_confusions,
    _resolve_condition,
)
from .._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    BuiltinKernelKey,
    CustomOperationStep,
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


def _expectation_confusions(
    factor_refs: tuple[RegisterRef, ...],
    resource_layout: ResourceLayout,
    noise_model: NoiseModel,
) -> tuple[Any, ...] | None:
    """Resolve readout confusion for a qubit expectation measurement."""
    return _resolve_confusions(
        factor_refs,
        ((0, 1),) * len(factor_refs),
        resource_layout,
        noise_model,
    )


def _build_expectation_tail(
    factor_axes: tuple[tuple[int, str], ...],
    factor_refs: tuple[RegisterRef, ...],
    *,
    scratch_start: int,
    resource_layout: ResourceLayout,
    noise_model: NoiseModel,
) -> tuple[ResolvedStep, ...]:
    """Build ideal basis changes followed by normally routed readout."""
    tail: list[ResolvedStep] = []
    for engine_index, letter in factor_axes:
        if letter == "X":
            tail.append(
                ApplyMatrixStep(
                    matrix=_H,
                    target_indices=(engine_index,),
                    kernel_key=BuiltinKernelKey.H,
                )
            )
        elif letter == "Y":
            tail.extend(
                (
                    ApplyMatrixStep(
                        matrix=_SDG,
                        target_indices=(engine_index,),
                        kernel_key=BuiltinKernelKey.SDG,
                    ),
                    ApplyMatrixStep(
                        matrix=_H,
                        target_indices=(engine_index,),
                        kernel_key=BuiltinKernelKey.H,
                    ),
                )
            )

    reported_digit_maps = ((0, 1),) * len(factor_axes)
    confusions = _expectation_confusions(
        factor_refs,
        resource_layout,
        noise_model,
    )
    tail.append(
        MeasurementStep(
            measured_indices=tuple(index for index, _letter in factor_axes),
            classical_indices=tuple(
                range(scratch_start, scratch_start + len(factor_axes))
            ),
            confusions=confusions,
            reported_digit_maps=(
                reported_digit_maps if confusions is not None else None
            ),
        )
    )
    return tuple(tail)


def _lower_reset(
    step: _AppliedOperation,
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
    step: _AppliedOperation,
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
    impl_map: MatrixImplementationMap,
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
        device_operands = tuple(
            resource_layout.device_label(target) for target in extent
        )
        channel_rule = _select_implementation(channel, device_operands, impl_map)
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
    step: _AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    impl_map: MatrixImplementationMap,
    noise_model: NoiseModel,
) -> list[ResolvedStep]:
    """Lower one registered operation and its attached channel noise.

    Structural resolution (device operands, engine indices, condition, rule
    selection) reads only the operation's type and targets, never its field
    values. A gate whose fields hold `Parameter`s therefore lowers here to an
    `ApplyParametricMatrixStep`, deferring the rule call and shape validation
    to sweep binding. Ordinary runs reject unbound programs before lowering.
    """
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
    steps: list[ResolvedStep] = []
    if isinstance(step.operation, ops.CustomOperation):
        steps.append(
            CustomOperationStep(
                operation=step.operation,
                implementation=rule,
                target_indices=engine_indices,
                condition=condition,
            )
        )
    elif _discover_parameters((step,)):
        steps.append(
            ApplyParametricMatrixStep(
                operation=step.operation,
                implementation=rule,
                targets=step.targets,
                target_indices=engine_indices,
                condition=condition,
            )
        )
    else:
        target_dims = tuple(engine_allocation.system_dims[i] for i in engine_indices)
        matrix = _build_matrix(
            rule, step.operation, targets=step.targets, target_dims=target_dims
        )
        steps.append(
            ApplyMatrixStep(
                matrix=matrix,
                target_indices=engine_indices,
                condition=condition,
                kernel_key=rule._kernel_key(step.operation, targets=step.targets),
            )
        )
    steps.extend(
        _lower_channels(
            type(step.operation),
            step.targets,
            condition,
            resource_layout,
            engine_allocation,
            noise_model,
            impl_map,
        )
    )
    return steps


def _inject_parameter_values(
    plan: Sequence[ResolvedStep],
    columns: Mapping[Parameter, np.ndarray],
) -> tuple[ResolvedStep, ...]:
    """Give each parametric step the value columns its `Parameter` fields read.

    Lowering stays independent of parameter values; a sweep injects its
    validated columns afterwards, so the plan an engine binds never refers to
    `Parameter` objects. Columns are shared by reference, not copied.
    """
    return tuple(
        (
            replace(
                step,
                values=tuple(
                    (field_info.name, columns[value])
                    for field_info in fields(step.operation)
                    if isinstance(
                        (value := getattr(step.operation, field_info.name)),
                        Parameter,
                    )
                ),
            )
            if isinstance(step, ApplyParametricMatrixStep)
            else step
        )
        for step in plan
    )
