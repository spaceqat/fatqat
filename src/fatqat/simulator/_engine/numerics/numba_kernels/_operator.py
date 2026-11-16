"""Numba local-matrix arithmetic on blocks of operator columns.

The caller selects the column range and supplies reusable scratch storage.
Offsets and complement strides are row-space values multiplied by the total
column count. Whole-plan sequencing and thread scheduling live in the workflow.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ._apply import _advance_base, _spread_base


@njit(cache=True)
def _apply_dense_columns(
    state, matrix, offsets, comp_strides, comp_dims, num_cosets, scratch, lo, hi
) -> None:  # pragma: no cover - compiled by Numba
    """Dense local matrix applied to columns ``[lo, hi)`` of every coset."""
    local_dim = offsets.shape[0]
    width = hi - lo
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(num_cosets):
        for r in range(local_dim):
            out = r * width
            for j in range(width):
                scratch[out + j] = 0.0 + 0.0j
            for c in range(local_dim):
                factor = matrix[r, c]
                src = base + offsets[c] + lo
                for j in range(width):
                    scratch[out + j] += factor * state[src + j]
        for r in range(local_dim):
            dst = base + offsets[r] + lo
            out = r * width
            for j in range(width):
                state[dst + j] = scratch[out + j]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_diagonal_columns(
    state, diagonal, offsets, comp_strides, comp_dims, num_cosets, scratch, lo, hi
) -> None:  # pragma: no cover - compiled by Numba
    """Diagonal application: scale in place, no gather and no scratch."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(num_cosets):
        for r in range(local_dim):
            factor = diagonal[r]
            dst = base + offsets[r] + lo
            for j in range(hi - lo):
                state[dst + j] *= factor
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_permutation_columns(
    state,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    scratch,
    lo,
    hi,
) -> None:  # pragma: no cover - compiled by Numba
    """Permutation application: one gather then one scaled scatter per coset."""
    local_dim = offsets.shape[0]
    width = hi - lo
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(num_cosets):
        for c in range(local_dim):
            src = base + offsets[c] + lo
            out = c * width
            for j in range(width):
                scratch[out + j] = state[src + j]
        for r in range(local_dim):
            factor = values[r]
            dst = base + offsets[r] + lo
            out = columns[r] * width
            for j in range(width):
                state[dst + j] = factor * scratch[out + j]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_sparse_columns(
    state,
    indptr,
    indices,
    data,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    scratch,
    lo,
    hi,
) -> None:  # pragma: no cover - compiled by Numba
    """CSR super-operator applied to columns ``[lo, hi)`` of every coset."""
    local_dim = offsets.shape[0]
    width = hi - lo
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    updated = local_dim * width
    for _ in range(num_cosets):
        for c in range(local_dim):
            src = base + offsets[c] + lo
            out = c * width
            for j in range(width):
                scratch[out + j] = state[src + j]
        for r in range(local_dim):
            out = updated + r * width
            for j in range(width):
                scratch[out + j] = 0.0 + 0.0j
            for k in range(indptr[r], indptr[r + 1]):
                factor = data[k]
                src = indices[k] * width
                for j in range(width):
                    scratch[out + j] += factor * scratch[src + j]
        for r in range(local_dim):
            dst = base + offsets[r] + lo
            out = updated + r * width
            for j in range(width):
                state[dst + j] = scratch[out + j]
        base = _advance_base(base, comp_strides, comp_dims, counter)
