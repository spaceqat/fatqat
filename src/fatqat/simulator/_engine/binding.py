"""Bind sweep parameter values into executable matrix plans.

A sweep lowers once. Parameter-holding gates stay `ApplyParametricMatrixStep`
values carrying one value column per `Parameter` field, and
`_bind_parameter_batch` turns the plan into one engine-ready step tuple per
batch entry. Every other step is shared by reference across entries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ..._backends.steps import (
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    ResolvedStep,
)
from ...errors import MatrixImplementationError
from ...implementation.matrix_map import (
    _BatchedBuilder,
    _build_matrix,
    _validate_matrix_shape,
)


def _bind_step(
    step: ApplyParametricMatrixStep,
    entry: int,
    target_dims: tuple[int, ...],
) -> ApplyMatrixStep:
    """Bind one batch entry into one step through the step's own matrix rule.

    `Program.assign_parameters` re-runs ``validate_targets`` when it rebuilds
    an applied operation, so a bound entry gives a parameter-dependent target
    check the same chance to reject the substituted value.
    """
    operation = replace(
        step.operation, **{name: column[entry] for name, column in step.values}
    )
    operation.validate_targets(step.targets)
    matrix = _build_matrix(
        step.implementation,
        operation,
        targets=step.targets,
        target_dims=target_dims,
    )
    return ApplyMatrixStep(
        matrix=matrix,
        target_indices=step.target_indices,
        condition=step.condition,
        kernel_key=step.implementation._kernel_key(
            step.operation, targets=step.targets
        ),
    )


def _build_matrix_batch(
    step: ApplyParametricMatrixStep,
    builder: _BatchedBuilder,
    target_dims: tuple[int, ...],
) -> np.ndarray:
    """Build one step's read-only ``(entries, d, d)`` matrix stack for the batch.

    Value columns are evaluated in float64, the package's working precision;
    a float64 column is used without copying. Builder failures raise
    `MatrixImplementationError` and a wrong-shaped stack raises
    `BackendValidationError`, as per-entry binding would.
    """
    template = step.operation
    try:
        columns = {
            name: np.asarray(column, dtype=np.float64) for name, column in step.values
        }

        def field(name: str) -> Any:
            return columns[name] if name in columns else getattr(template, name)

        stack = builder(field, step.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(template).__name__} raised: {exc}"
        ) from exc
    _validate_matrix_shape(stack.shape[1:], template, target_dims)
    stack.flags.writeable = False
    return stack


def _bind_parameter_batch(
    plan: Sequence[ResolvedStep],
    system_dims: Sequence[int],
) -> Iterator[tuple[ResolvedStep, ...]]:
    """Bind a plan for every entry of its parameter batch, then yield each one.

    The batch size is the length of the injected value columns; a plan without
    parametric steps is a batch of one. Built-in parametric gates get every
    entry's matrices at once from their batched rules; every other parametric
    step calls its rule once per entry. Each entry's bound steps are handed
    out once, so an entry is released as soon as its caller drops it.
    """
    parametric = {
        index: step
        for index, step in enumerate(plan)
        if isinstance(step, ApplyParametricMatrixStep)
    }
    if any(not step.values for step in parametric.values()):
        raise RuntimeError("parameter values must be injected before binding")
    batch_size = min(
        (len(step.values[0][1]) for step in parametric.values()), default=1
    )
    bound_steps: dict[int, deque[ApplyMatrixStep]] = {}
    for index, step in parametric.items():
        target_dims = tuple(system_dims[i] for i in step.target_indices)
        builder = step.implementation._batch_builder(step.operation)
        if builder is None:
            bound_steps[index] = deque(
                _bind_step(step, entry, target_dims) for entry in range(batch_size)
            )
            continue
        kernel_key = step.implementation._kernel_key(
            step.operation, targets=step.targets
        )
        bound_steps[index] = deque(
            ApplyMatrixStep(
                matrix=matrix,
                target_indices=step.target_indices,
                condition=step.condition,
                kernel_key=kernel_key,
            )
            for matrix in _build_matrix_batch(step, builder, target_dims)
        )
    for _ in range(batch_size):
        yield tuple(
            bound_steps[index].popleft() if index in bound_steps else step
            for index, step in enumerate(plan)
        )
