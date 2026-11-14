"""Step-level gate fusion for execution plans.

`schedule` splits a plan into blocks of adjacent gate steps that one matrix
can replace. It reads step types, conditions, targets, dimensions, and kernel
keys, never matrices, so a lowered plan and each of its bound plans get the
same blocks. `fuse_blocks` then builds one matrix step for each block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import prod
from typing import TypeGuard, cast

from ...._backends.steps import (
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    ResolvedStep,
)
from ..._execution_contract import _GateFusionLimits
from ..numerics._gate_matrices import _DENSE, _KERNEL_SPECS, _embed_dense


def schedule(
    plan: Sequence[ResolvedStep],
    system_dims: Sequence[int],
    limits: _GateFusionLimits,
) -> tuple[range, ...]:
    """Split ``plan`` into consecutive blocks of mergeable gate steps.

    The ranges cover ``range(len(plan))`` in order, and each step index is in
    exactly one range. Below ``limits.min_size`` every step is its own block.
    Otherwise one linear scan keeps at most one open block. A gate joins the
    open block only when all of these hold:

    1. The merged targets span a local dimension of at most
       ``limits.max_dim``.
    2. With a cost table, the merged pass is strictly cheaper than the two
       separate passes.
    3. With ``limits.structure_aware``, the gate and the block are both dense
       or both not dense.

    Every other step is a barrier. A barrier closes the open block and forms a
    block of its own, whatever its targets.
    """
    if prod(system_dims) < limits.min_size:
        return tuple(range(index, index + 1) for index in range(len(plan)))
    cost = None if limits.cost_by_dim is None else dict(limits.cost_by_dim)
    blocks: list[range] = []
    start: int | None = None
    union: frozenset[int] = frozenset()
    dense = False
    for index, step in enumerate(plan):
        if not _is_gate(step):
            if start is not None:
                blocks.append(range(start, index))
                start = None
            blocks.append(range(index, index + 1))
            continue
        targets = frozenset(step.target_indices)
        step_dense = _structure(step) == _DENSE
        if start is not None:
            merged = union | targets
            merged_dim = _local_dim(merged, system_dims)
            if (
                merged_dim <= limits.max_dim
                and _cheaper(
                    cost,
                    merged_dim,
                    _local_dim(union, system_dims),
                    _local_dim(targets, system_dims),
                )
                and (not limits.structure_aware or dense == step_dense)
            ):
                union = merged
                continue
            blocks.append(range(start, index))
        start, union, dense = index, targets, step_dense
    if start is not None:
        blocks.append(range(start, len(plan)))
    return tuple(blocks)


def fuse_blocks(
    plan: Sequence[ResolvedStep],
    blocks: Sequence[range],
    system_dims: Sequence[int],
) -> tuple[ResolvedStep, ...]:
    """Replace each block of ``plan`` with one step.

    A one-step block keeps its original step object. A longer block becomes
    one unconditional `ApplyMatrixStep` on the ascending union of its targets.
    Its matrix is the product of the member matrices, with later steps on the
    left. The fused step has no kernel key, so engines classify its matrix.

    Raises:
        RuntimeError: ``plan`` holds a parametric step. The check covers the
            whole plan, not only multi-step blocks.
    """
    if any(isinstance(step, ApplyParametricMatrixStep) for step in plan):
        raise RuntimeError(
            "Gate fusion requires a bound plan. "
            "Bind every parametric step before fusion."
        )
    # `schedule` puts only gates in multi-step blocks, and parametric steps
    # were rejected above, so every multi-step block holds matrix steps.
    return tuple(
        (
            plan[block.start]
            if len(block) == 1
            else _fuse(
                cast("list[ApplyMatrixStep]", [plan[i] for i in block]), system_dims
            )
        )
        for block in blocks
    )


def _is_gate(
    step: ResolvedStep,
) -> TypeGuard[ApplyMatrixStep | ApplyParametricMatrixStep]:
    """Whether ``step`` is an unconditional matrix step that fusion may merge."""
    return (
        isinstance(step, (ApplyMatrixStep, ApplyParametricMatrixStep))
        and step.condition is None
    )


def _structure(step: ApplyMatrixStep | ApplyParametricMatrixStep) -> int:
    """Declared structure code of a gate step. An undeclared key is dense.

    A parametric step derives its key with the expression that binding uses,
    so binding does not change the code.
    """
    if isinstance(step, ApplyParametricMatrixStep):
        key = step.implementation._kernel_key(step.operation, targets=step.targets)
    else:
        key = step.kernel_key
    return _DENSE if key is None else _KERNEL_SPECS.get(key, _DENSE)


def _local_dim(targets: Iterable[int], system_dims: Sequence[int]) -> int:
    """Local Hilbert dimension spanned by ``targets``."""
    return prod(system_dims[target] for target in targets)


def _cheaper(cost: dict[int, float] | None, merged: int, block: int, gate: int) -> bool:
    """Whether one pass at ``merged`` costs less than passes at ``block`` and ``gate``.

    Without a cost table every merge within the width limit is accepted. A
    dimension missing from the table blocks the merge.
    """
    if cost is None:
        return True
    if merged not in cost or block not in cost or gate not in cost:
        return False
    return cost[merged] < cost[block] + cost[gate]


def _fuse(
    gates: Sequence[ApplyMatrixStep], system_dims: Sequence[int]
) -> ApplyMatrixStep:
    """One matrix step equal to applying ``gates`` in order."""
    union = tuple(sorted({target for gate in gates for target in gate.target_indices}))
    matrices = [
        (
            gate.matrix
            if gate.target_indices == union
            else _embed_dense(gate.matrix, gate.target_indices, union, system_dims)
        )
        for gate in gates
    ]
    product = matrices[0]
    for matrix in matrices[1:]:
        product = matrix @ product
    return ApplyMatrixStep(matrix=product, target_indices=union)
