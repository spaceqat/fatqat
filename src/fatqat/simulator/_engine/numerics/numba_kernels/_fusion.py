"""Numerical gate/channel and operator fusion for Numba engines.

Two distinct passes are enabled by ``fusion=True``:

* ``_fuse_gate_channels`` rewrites resolved steps: an unconditional gate and
  immediately following unconditional channel on identical targets become one
  channel. There is no whole-system size threshold for this pass.
* ``_fuse_operator_payloads`` combines adjacent numerical operators, including
  gates. Operator engines invoke it only above ``_MIN_SIZE_TO_FUSE``; its cost
  and width limits then decide which neighbors to merge. CSR payloads are
  barriers. This pass preserves order and does not commute operations.

Usage by execution route (when fusion is enabled):

    Runtime / representation     Workflow              Fusion passes
    NumPy / all                  Python                Neither
    Numba / statevector          Python or compiled SV Neither
    Numba / density matrix       Python                Neither
    Numba / unitary              Compiled operator     Operator payloads only
    Numba / superoperator        Compiled operator     Both, in that order

State-evolution fusion (SV and DM) is deferred to a future implementation.
Process-based shots therefore perform no fusion in either parent or workers.

For compiled superoperator execution, gate/channel fusion precedes construction
of numerical payloads, then operator fusion precedes packing. Compiled unitary
execution uses only the latter pass. Neither pass runs inside the execution loop
or per shot. Compiled SV trajectories do not currently use either pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
from numba import njit

from ....._backends.steps import ApplyChannelStep, ApplyMatrixStep
from ._apply import _DENSE, _DIAGONAL, _PERMUTATION

# --- gate/channel fusion ---


def _fuse_gate_channels(plan: list) -> list:
    """Merge each unconditional gate directly followed by an unconditional
    channel on the same targets into the one channel it equals.

    ``channel(gate(rho)) = sum_i K_i (M rho M^dagger) K_i^dagger =
    sum_i (K_i M) rho (K_i M)^dagger`` is itself a Kraus channel, so the pair
    collapses to a single `ApplyChannelStep` with operators ``{K_i M}`` - one
    super-operator pass over ``vec(rho)`` instead of two. Only exact-adjacent,
    identically-targeted, unconditional pairs merge; every other case is left
    untouched. The composition is exact, so results match the two-pass form to
    floating-point round-off (density-matrix reproducibility is allclose, not
    bit-identical). Density-matrix only - a gate is deterministic there, so
    folding it into the following channel never perturbs the RNG stream.
    """
    fused: list = []
    i = 0
    while i < len(plan):
        step = plan[i]
        nxt = plan[i + 1] if i + 1 < len(plan) else None
        if (
            isinstance(step, ApplyMatrixStep)
            and step.condition is None
            and isinstance(nxt, ApplyChannelStep)
            and nxt.condition is None
            and nxt.target_indices == step.target_indices
        ):
            merged = tuple(kraus @ step.matrix for kraus in nxt.kraus_ops)
            fused.append(
                ApplyChannelStep(kraus_ops=merged, target_indices=step.target_indices)
            )
            i += 2
        else:
            fused.append(step)
            i += 1
    return fused


# --- operator fusion policy and state ---
#
# Merging adjacent steps into one wider step trades passes for arithmetic. The
# dense kernel's cost is ~linear in the local dimension, while the diagonal and
# permutation kernels are flat in it, and monomial matrices (at most one nonzero
# per row - `_classify_matrix`'s _DIAGONAL/_PERMUTATION) are closed under
# multiplication. So a run of phase or permutation gates collapses into one pass
# at any width, while dense runs merge only while `_pass_cost` says the wider
# pass is cheaper. Fusion multiplies matrices together, so it equals the unfused
# plan to floating-point tolerance, not bit for bit.

# Local dimension at which dense compute overtakes memory, setting `_pass_cost`.
_DENSE_COST_KNEE = 6.0
# Operator size below which fusion's plan-preparation cost outweighs the passes
# it saves.
_MIN_SIZE_TO_FUSE = 1 << 18
# Widest merged operator, per structure. Monomials are capped by the bookkeeping
# each merge rebuilds; dense is a backstop the cost model stops well short of.
_MAX_FUSED_MONOMIAL_DIM = 1 << 12
_MAX_FUSED_DENSE_DIM = 1 << 6


def _pass_cost(code: int, local_dim: int) -> float:
    """Relative cost of one pass, normalized to a 1-qubit dense pass."""
    if code == _DENSE:
        return max(1.0, local_dim / _DENSE_COST_KNEE)
    return 1.0


@dataclass
class _FusionBlock:
    """A run of adjacent payloads accumulated into one operator.

    ``code`` is the accumulated structure: monomial while every merged step was
    monomial, dense from the first dense one onwards.
    """

    targets: tuple[int, ...]
    code: int
    columns: np.ndarray | None
    values: np.ndarray | None
    matrix: np.ndarray | None
    merged: int

    @property
    def local_dim(self) -> int:
        return len(self.columns) if self.code != _DENSE else len(self.matrix)

    def cost(self) -> float:
        return _pass_cost(self.code, self.local_dim)


# --- operator embedding and composition ---


def _local_places(dims: Sequence[int]) -> list[int]:
    """Mixed-radix place values with element 0 most significant."""
    places = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        places[i] = places[i + 1] * dims[i + 1]
    return places


def _embedding(
    from_targets: tuple[int, ...], to_targets: tuple[int, ...], dims: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index maps embedding a ``from_targets`` operator into ``to_targets``.

    Returns ``(local, base, offsets)``: ``local[x]`` is the narrow row wide
    index ``x`` selects, ``base[x]`` is ``x`` with the narrow digits cleared,
    and ``offsets[r]`` is the wide offset of narrow index ``r``. Both index
    spaces are most-significant-first in their own target order.
    """
    to_dims = [dims[t] for t in to_targets]
    from_dims = [dims[t] for t in from_targets]
    to_places = _local_places(to_dims)
    from_places = _local_places(from_dims)
    positions = [to_targets.index(t) for t in from_targets]

    wide = np.arange(prod(to_dims), dtype=np.int64)
    narrow = np.arange(prod(from_dims), dtype=np.int64)
    local = np.zeros_like(wide)
    offsets = np.zeros_like(narrow)
    for i, position in enumerate(positions):
        local += ((wide // to_places[position]) % to_dims[position]) * from_places[i]
        offsets += ((narrow // from_places[i]) % from_dims[i]) * to_places[position]
    return local, wide - offsets[local], offsets


def _monomial_operand(
    targets: tuple[int, ...], union: tuple[int, ...], dims: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descriptor letting `_compose_monomials` embed ``targets`` into ``union``.

    Returns ``(positions, places, offsets)``: where each subsystem sits in
    ``union``, its own place values, and the wide offset of each narrow index.
    """
    narrow_dims = [dims[t] for t in targets]
    places = _local_places(narrow_dims)
    positions = [union.index(t) for t in targets]
    to_places = _local_places([dims[t] for t in union])
    narrow = np.arange(prod(narrow_dims), dtype=np.int64)
    offsets = np.zeros_like(narrow)
    for i, position in enumerate(positions):
        offsets += ((narrow // places[i]) % narrow_dims[i]) * to_places[position]
    return (
        np.asarray(positions, dtype=np.int64),
        np.asarray(places, dtype=np.int64),
        offsets,
    )


@njit(cache=True)
def _compose_monomials(
    b_columns,
    b_values,
    b_positions,
    b_places,
    b_offsets,
    s_columns,
    s_values,
    s_positions,
    s_places,
    s_offsets,
    to_places,
    to_dims,
) -> tuple:  # pragma: no cover - compiled by Numba
    """Embed two monomials onto a shared index space and multiply them.

    ``b`` runs first and ``s`` second, so this builds ``s . b``.
    """
    size = 1
    for i in range(to_dims.shape[0]):
        size *= to_dims[i]
    out_columns = np.empty(size, dtype=np.int64)
    out_values = np.empty(size, dtype=np.complex128)
    for x in range(size):
        row_s = 0
        for i in range(s_positions.shape[0]):
            p = s_positions[i]
            row_s += ((x // to_places[p]) % to_dims[p]) * s_places[i]
        # Clear this operand's digits from x, then set the ones it maps to.
        y = x - s_offsets[row_s] + s_offsets[s_columns[row_s]]
        row_b = 0
        for i in range(b_positions.shape[0]):
            p = b_positions[i]
            row_b += ((y // to_places[p]) % to_dims[p]) * b_places[i]
        out_columns[x] = y - b_offsets[row_b] + b_offsets[b_columns[row_b]]
        out_values[x] = s_values[row_s] * b_values[row_b]
    return out_columns, out_values


def _embed_dense(
    matrix: np.ndarray,
    from_targets: tuple[int, ...],
    to_targets: tuple[int, ...],
    dims: Sequence[int],
) -> np.ndarray:
    """Widen a dense matrix onto ``to_targets``, tensoring identity onto the rest."""
    local, base, offsets = _embedding(from_targets, to_targets, dims)
    size = len(local)
    out = np.zeros((size, size), dtype=np.complex128)
    rows = np.arange(size, dtype=np.int64)
    for column, offset in enumerate(offsets):
        out[rows, base + offset] = matrix[local, column]
    return out


def _widen_dense(
    block: _FusionBlock, union: tuple[int, ...], dims: Sequence[int]
) -> np.ndarray:
    """A block's dense matrix on ``union``, materializing a monomial if needed."""
    if block.code == _DENSE:
        matrix = block.matrix
    else:
        matrix = np.zeros((len(block.columns),) * 2, dtype=np.complex128)
        matrix[np.arange(len(block.columns)), block.columns] = block.values
    if block.targets == union:
        return matrix
    return _embed_dense(matrix, block.targets, union, dims)


# --- operator fusion assembly ---


def _payload_block(payload: tuple) -> _FusionBlock:
    """Start a block from one payload."""
    matrix, code, columns, values, _sparse, targets = payload
    monomial = code in (_DIAGONAL, _PERMUTATION)
    local_dim = len(columns) if monomial else len(matrix)
    return _FusionBlock(
        targets=targets,
        code=code,
        columns=np.asarray(columns[:local_dim], dtype=np.int64) if monomial else None,
        values=(
            np.asarray(values[:local_dim], dtype=np.complex128) if monomial else None
        ),
        matrix=None if monomial else np.asarray(matrix, dtype=np.complex128),
        merged=1,
    )


def _block_payload(block: _FusionBlock, original: tuple) -> tuple:
    """Convert a block back into a payload, or hand back the untouched original."""
    if block.merged == 1:
        return original
    if block.code == _DENSE:
        return (block.matrix, _DENSE, None, None, None, block.targets)
    # A monomial whose columns never move reaches the cheaper in-place kernel.
    identity = np.array_equal(block.columns, np.arange(len(block.columns)))
    code = _DIAGONAL if identity else _PERMUTATION
    return (None, code, block.columns, block.values, None, block.targets)


def _merge_block(
    block: _FusionBlock, payload: tuple, dims: Sequence[int]
) -> _FusionBlock | None:
    """Merge ``payload`` after ``block``, or return ``None`` to keep them apart.

    Accepted only when the combined pass is strictly cheaper than the two it
    replaces. Caps are checked before anything is materialized.
    """
    _matrix, code, _columns, _values, _sparse, targets = payload
    # Sorting defines a new private embedding space only. The helpers below
    # re-embed each original positional target tuple into that union, so the
    # union never replaces a local matrix's target order.
    union = tuple(sorted(set(block.targets) | set(targets)))
    local_dim = prod(dims[t] for t in union)
    monomial = block.code != _DENSE and code in (_DIAGONAL, _PERMUTATION)
    merged_code = _PERMUTATION if monomial else _DENSE
    cap = _MAX_FUSED_MONOMIAL_DIM if monomial else _MAX_FUSED_DENSE_DIM
    if local_dim > cap:
        return None
    apart = block.cost() + _pass_cost(code, prod(dims[t] for t in targets))
    if _pass_cost(merged_code, local_dim) >= apart:
        return None

    step = _payload_block(payload)
    if monomial:
        to_dims = np.asarray([dims[t] for t in union], dtype=np.int64)
        to_places = np.asarray(_local_places(list(to_dims)), dtype=np.int64)
        columns, values = _compose_monomials(
            block.columns,
            block.values,
            *_monomial_operand(block.targets, union, dims),
            step.columns,
            step.values,
            *_monomial_operand(step.targets, union, dims),
            to_places,
            to_dims,
        )
        return _FusionBlock(
            targets=union,
            code=_PERMUTATION,
            columns=columns,
            values=values,
            matrix=None,
            merged=block.merged + 1,
        )
    return _FusionBlock(
        targets=union,
        code=_DENSE,
        columns=None,
        values=None,
        matrix=_widen_dense(step, union, dims) @ _widen_dense(block, union, dims),
        merged=block.merged + 1,
    )


def _fuse_operator_payloads(payloads: list[tuple], dims: Sequence[int]) -> list[tuple]:
    """Merge adjacent operator payloads into wider ones where that is cheaper.

    A greedy left-to-right pass over payloads in execution order; only adjacent
    payloads merge, so no commutation analysis is involved. A CSR step is a
    barrier on both sides, since `_pass_cost` cannot see its skipped zeros.
    """
    fused: list[tuple] = []
    block: _FusionBlock | None = None
    source: tuple | None = None
    for payload in payloads:
        if payload[4] is not None:
            if block is not None:
                fused.append(_block_payload(block, source))
                block = source = None
            fused.append(payload)
            continue
        if block is None:
            block, source = _payload_block(payload), payload
            continue
        merged = _merge_block(block, payload, dims)
        if merged is None:
            fused.append(_block_payload(block, source))
            block, source = _payload_block(payload), payload
        else:
            block = merged
    if block is not None:
        fused.append(_block_payload(block, source))
    return fused
