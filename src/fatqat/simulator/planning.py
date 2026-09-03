"""Private lowering helpers for the matrix-family simulator backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from math import prod
from numbers import Real
from typing import Any, cast

from .._index_allocation import _ClassicalAllocation, _EngineAllocation
from .._parameter_binding import _parameter_field_slots
from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
)
from ..implementation import MatrixImplementation, MatrixImplementationMap
from ..implementation._operation_registry import _select_implementation
from ..implementation.matrices import _H, _SDG
from ..noise import ChannelImplementationMap, NoiseModel
from ..noise.base import _validate_kraus_shapes
from ..operations import Measurement, Operation, PulseOperation
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
    BuiltinKernelKey,
    LossStep,
    MeasurementStep,
    ResetStep,
    PutStep,
    ResolvedStep,
)


@dataclass(frozen=True)
class _MatrixRecipe:
    """Deferred local matrix for a gate whose fields still hold `Parameter`s.

    Not a `ResolvedStep`: an engine never receives one. A recipe records
    everything lowering resolved structurally - rule selection, engine
    indices, condition, kernel identity, expected shape - and defers only the
    numeric matrix to sweep replay, when `_ParametricPlan.materialize`
    substitutes one θ vector and turns it into an ordinary `ApplyMatrixStep`.

    The fields form two halves. ``kernel_key``, ``param_slots``,
    ``target_indices``, ``target_dims`` and ``condition`` are plain data an
    engine could read directly. ``rule``, ``operation_template`` and
    ``targets`` are the Python fallback: the same registered rule the
    fully-bound path calls, so matrix math keeps one source of truth, plus the
    program targets the rule protocol (``rule(op, *, targets)``) needs for
    dimension context.

    Attributes:
        rule: The matrix implementation selected at lowering.
        operation_template: The operation as declared, with `Parameter`
            objects still in its fields.
        param_slots: θ-vector positions consumed by the template's
            `Parameter` fields, in dataclass field order.
        targets: The operation's program targets, passed to ``rule`` at
            replay.
        target_indices: Flat subsystem indices the resolved matrix acts on.
        target_dims: Local dimensions of the addressed subsystems, for
            replay-time shape validation.
        condition: Lowered feedforward guard, or ``None``.
        kernel_key: Canonical identity of the selected implementation; a
            structural constant per registration.
    """

    rule: MatrixImplementation
    operation_template: Operation
    param_slots: tuple[int, ...]
    targets: tuple[RegisterRef, ...]
    target_indices: tuple[int, ...]
    target_dims: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None
    kernel_key: BuiltinKernelKey | None = None


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
    step: _AppliedOperation,
    resource_layout: ResourceLayout,
    engine_allocation: _EngineAllocation,
    classical_allocation: _ClassicalAllocation,
    impl_map: MatrixImplementationMap,
    noise_model: NoiseModel,
    channel_map: ChannelImplementationMap,
    *,
    param_order: tuple[Parameter, ...] | None = None,
) -> list[ResolvedStep | _MatrixRecipe]:
    """Lower one ordinary-gate operation and its attached channel noise.

    Structural resolution (device operands, engine indices, condition, rule
    selection, kernel identity, target dimensions) reads only the operation's
    type and targets, never its field values. A gate whose fields hold
    `Parameter`s therefore lowers here without a concrete matrix: when
    ``param_order`` supplies the sweep's θ order it becomes a
    `_MatrixRecipe`, deferring the rule call and shape validation to
    replay. Without ``param_order`` (an ordinary run) or without parameter
    fields, the fully-bound path is exactly today's behavior.
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
    target_dims = tuple(engine_allocation.system_dims[i] for i in engine_indices)
    param_slots = _parameter_field_slots(step.operation, param_order)
    steps: list[ResolvedStep | _MatrixRecipe] = []
    if param_slots is not None:
        steps.append(
            _MatrixRecipe(
                rule=rule,
                operation_template=step.operation,
                param_slots=param_slots,
                targets=step.targets,
                target_indices=engine_indices,
                target_dims=target_dims,
                condition=condition,
                kernel_key=rule._kernel_key(step.operation, targets=step.targets),
            )
        )
    else:
        try:
            matrix = rule(step.operation, targets=step.targets)
        except Exception as exc:
            raise MatrixImplementationError(
                f"implementation for {type(step.operation).__name__} raised: {exc}"
            ) from exc

        expected = prod(target_dims)
        if matrix.shape != (expected, expected):
            raise BackendValidationError(
                f"{type(step.operation).__name__} resolved to a "
                f"{matrix.shape} matrix, incompatible with target "
                f"dimensions {target_dims} (expected "
                f"{(expected, expected)})"
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
            channel_map,
        )
    )
    return steps


def _materialize_recipe(
    recipe: _MatrixRecipe, theta: Sequence[Real]
) -> ApplyMatrixStep:
    """Realize one recipe into a concrete step for one θ vector.

    Substitutes the θ values the recipe's ``param_slots`` point at into the
    template's `Parameter` fields, re-runs the operation's ``validate_targets``
    hook, then calls the rule with the same error wrapping and shape
    validation the fully-bound lowering path applies.
    Failures raise directly, exactly as they raise from lowering in ``run()``.
    """
    replacements: dict[str, Real] = {}
    slot_index = 0
    for field_info in fields(recipe.operation_template):
        value = getattr(recipe.operation_template, field_info.name)
        if isinstance(value, Parameter):
            replacements[field_info.name] = theta[recipe.param_slots[slot_index]]
            slot_index += 1
    operation = replace(recipe.operation_template, **replacements)
    # Binding through Program re-runs this hook when it rebuilds the applied
    # operation; a replayed recipe must give a parameter-dependent target
    # check the same chance to reject the substituted value.
    operation.validate_targets(recipe.targets)
    try:
        matrix = recipe.rule(operation, targets=recipe.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(operation).__name__} raised: {exc}"
        ) from exc
    expected = prod(recipe.target_dims)
    if matrix.shape != (expected, expected):
        raise BackendValidationError(
            f"{type(operation).__name__} resolved to a "
            f"{matrix.shape} matrix, incompatible with target "
            f"dimensions {recipe.target_dims} (expected "
            f"{(expected, expected)})"
        )
    return ApplyMatrixStep(
        matrix=matrix,
        target_indices=recipe.target_indices,
        condition=recipe.condition,
        kernel_key=recipe.kernel_key,
    )


@dataclass(frozen=True)
class _ParametricPlan:
    """A lowered plan whose parameter-holding gates are still recipes.

    The sweep counterpart of a ``tuple[ResolvedStep, ...]``, kept as its own
    type so the `ResolvedStep` contract ("an engine can execute every member")
    stays true. Structural lowering happened once; ``materialize`` turns the
    plan into an engine-ready step tuple for one θ vector. Non-parametric
    steps are shared by reference across rows (their payloads are frozen).

    Attributes:
        steps: Lowered steps in program order; a recipe stands where its
            `ApplyMatrixStep` will go.
        param_order: The flat θ order (first-appearance discovery order)
            that every recipe's ``param_slots`` index into.
    """

    steps: tuple[ResolvedStep | _MatrixRecipe, ...]
    param_order: tuple[Parameter, ...]

    def materialize(self, theta: Sequence[Real]) -> tuple[ResolvedStep, ...]:
        """Return the concrete, engine-ready plan for one θ vector."""
        return tuple(
            (
                _materialize_recipe(step, theta)
                if isinstance(step, _MatrixRecipe)
                else step
            )
            for step in self.steps
        )
