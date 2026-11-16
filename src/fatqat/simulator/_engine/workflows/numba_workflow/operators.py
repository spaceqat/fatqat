"""Compiled whole-plan operator workflows over independent column blocks."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit, prange

from ..._execution_policy import _ExecutionPolicy as ExecutionPolicy
from ..._numba_scope import _kernel_thread_workers
from ...numerics.numba_kernels._apply import _DENSE, _DIAGONAL, _PERMUTATION, _SPARSE
from ...numerics.numba_kernels._operator import (
    _apply_dense_columns,
    _apply_diagonal_columns,
    _apply_permutation_columns,
    _apply_sparse_columns,
)
from ...numerics.numba_kernels._layouts import _build_apply_layout


class _NumbaOperatorExecution(NamedTuple):
    """Packed operator plan and launch dimensions; only arrays enter Numba."""

    packed_plan: tuple[np.ndarray, ...] | None
    scratch_rows: int
    n_columns: int
    n_chunks: int


def _execute_operator_plan(
    state: np.ndarray, payload: _NumbaOperatorExecution
) -> np.ndarray:
    """Run a packed operator plan on initialized storage in the caller's scope."""
    if payload.packed_plan is not None:
        operator = np.ascontiguousarray(state, dtype=np.complex128)
        flat = operator.reshape(-1)
        if payload.n_chunks > 1:
            _run_operator_plan_parallel(
                flat,
                payload.n_columns,
                payload.packed_plan,
                payload.scratch_rows,
                payload.n_chunks,
            )
        else:
            _run_operator_plan_serial(
                flat,
                payload.n_columns,
                payload.packed_plan,
                payload.scratch_rows,
            )
        return flat.reshape(operator.shape)
    return state


# --- compiled whole-plan operator kernels (one parallel region) ---
#
# An operator buffer is ``(rows, columns)`` row-major and every step acts on the
# row index alone, so a block of columns is closed under the whole plan and the
# parallel region sits outside the step loop. Offsets and complement strides are
# row-space values pre-multiplied by ``columns``, so ``base + offsets[c] + lo``
# addresses the flat buffer directly.


@njit(cache=True)
def _run_operator_steps(
    state, plan, lo, hi, scratch
) -> None:  # pragma: no cover - compiled by Numba
    """Apply every step of a packed operator plan to columns ``[lo, hi)``."""
    (
        code,
        mat_flat,
        mat_ptr,
        local_dims,
        sv_flat_columns,
        sv_flat_values,
        sv_ptr,
        off_flat,
        off_ptr,
        comp_stride_flat,
        comp_dim_flat,
        comp_ptr,
        num_cosets,
        sp_indptr_flat,
        sp_indices_flat,
        sp_data_flat,
        sp_indptr_ptr,
        sp_data_ptr,
    ) = plan
    for s in range(code.shape[0]):
        local_dim = local_dims[s]
        offsets = off_flat[off_ptr[s] : off_ptr[s + 1]]
        strides = comp_stride_flat[comp_ptr[s] : comp_ptr[s + 1]]
        comp_dims = comp_dim_flat[comp_ptr[s] : comp_ptr[s + 1]]
        cosets = num_cosets[s]
        step_code = code[s]
        if step_code == _DIAGONAL:
            values = sv_flat_values[sv_ptr[s] : sv_ptr[s + 1]]
            _apply_diagonal_columns(
                state, values, offsets, strides, comp_dims, cosets, scratch, lo, hi
            )
        elif step_code == _PERMUTATION:
            cols = sv_flat_columns[sv_ptr[s] : sv_ptr[s + 1]]
            values = sv_flat_values[sv_ptr[s] : sv_ptr[s + 1]]
            _apply_permutation_columns(
                state,
                cols,
                values,
                offsets,
                strides,
                comp_dims,
                cosets,
                scratch,
                lo,
                hi,
            )
        elif step_code == _SPARSE:
            indptr = sp_indptr_flat[sp_indptr_ptr[s] : sp_indptr_ptr[s + 1]]
            indices = sp_indices_flat[sp_data_ptr[s] : sp_data_ptr[s + 1]]
            data = sp_data_flat[sp_data_ptr[s] : sp_data_ptr[s + 1]]
            _apply_sparse_columns(
                state,
                indptr,
                indices,
                data,
                offsets,
                strides,
                comp_dims,
                cosets,
                scratch,
                lo,
                hi,
            )
        else:
            matrix = mat_flat[mat_ptr[s] : mat_ptr[s + 1]].reshape(local_dim, local_dim)
            _apply_dense_columns(
                state, matrix, offsets, strides, comp_dims, cosets, scratch, lo, hi
            )


@njit(cache=True)
def _run_operator_plan_serial(
    state, n_columns, plan, scratch_rows
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Whole-plan application on one thread (no parallel region entered)."""
    scratch = np.empty(scratch_rows * n_columns, dtype=np.complex128)
    _run_operator_steps(state, plan, 0, n_columns, scratch)
    return state


@njit(cache=True, parallel=True)
def _run_operator_plan_parallel(
    state, n_columns, plan, scratch_rows, n_chunks
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Whole-plan application, split over column blocks in ONE parallel region."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        lo = chunk * n_columns // n_chunks
        hi = (chunk + 1) * n_columns // n_chunks
        if lo < hi:
            scratch = np.empty(scratch_rows * (hi - lo), dtype=np.complex128)
            _run_operator_steps(state, plan, lo, hi, scratch)
    return state


# Minimum per-thread work before a compiled whole-plan run goes parallel.
_GRAIN_TO_THREAD_OPERATOR = 1 << 14


def _operator_chunks(
    n_steps: int, size: int, n_columns: int, policy: ExecutionPolicy
) -> int:
    """Column-block count selected by the resolved local thread policy."""
    max_workers = _kernel_thread_workers(policy)
    if policy.kernel_strategy == "serial" or max_workers <= 1:
        return 1
    if (
        policy.kernel_strategy != "threads"
        and n_steps * size < max_workers * _GRAIN_TO_THREAD_OPERATOR
    ):
        return 1
    return max(1, min(max_workers, n_columns))


def _pack_operator_plan(
    payloads: list[tuple], row_dims: tuple[int, ...], n_columns: int
) -> tuple[tuple, int]:
    """Flatten operator payloads into `_run_operator_steps`'s arrays.

    Each payload is ``(matrix, code, columns, values, sparse, row_targets)``.
    Ragged per-step data is concatenated with ``*_ptr`` index arrays; row-space
    offsets and strides are pre-multiplied by ``n_columns``.

    Returns the packed tuple and the scratch row count the widest step needs.
    """
    n_steps = len(payloads)
    code = np.empty(n_steps, dtype=np.int64)
    local_dims = np.empty(n_steps, dtype=np.int64)
    num_cosets = np.empty(n_steps, dtype=np.int64)
    mat_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sv_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    off_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    comp_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sp_indptr_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sp_data_ptr = np.zeros(n_steps + 1, dtype=np.int64)

    mat_parts: list[np.ndarray] = []
    sv_column_parts: list[np.ndarray] = []
    sv_value_parts: list[np.ndarray] = []
    off_parts: list[np.ndarray] = []
    comp_stride_parts: list[np.ndarray] = []
    comp_dim_parts: list[np.ndarray] = []
    sp_indptr_parts: list[np.ndarray] = []
    sp_index_parts: list[np.ndarray] = []
    sp_data_parts: list[np.ndarray] = []
    scratch_rows = 1

    for s, (matrix, step_code, columns, values, sparse, targets) in enumerate(payloads):
        offsets, strides, dims, cosets, _ = _build_apply_layout(row_dims, targets)
        local_dim = int(offsets.shape[0])
        off_parts.append(offsets * n_columns)
        comp_stride_parts.append(strides * n_columns)
        comp_dim_parts.append(dims)
        local_dims[s] = local_dim
        num_cosets[s] = cosets

        if sparse is not None:
            step_code = _SPARSE
        code[s] = step_code
        if step_code == _SPARSE:
            indptr, indices, data = sparse
            sp_indptr_parts.append(np.asarray(indptr, dtype=np.int64))
            sp_index_parts.append(np.asarray(indices, dtype=np.int64))
            sp_data_parts.append(np.asarray(data, dtype=np.complex128))
            # Gather block plus accumulator block.
            scratch_rows = max(scratch_rows, 2 * local_dim)
        else:
            scratch_rows = max(scratch_rows, local_dim)
            if step_code == _DENSE:
                mat_parts.append(np.asarray(matrix, dtype=np.complex128).reshape(-1))
            else:
                sv_column_parts.append(np.asarray(columns, dtype=np.int64))
                sv_value_parts.append(np.asarray(values, dtype=np.complex128))

        dense_step = step_code == _DENSE
        sparse_step = step_code == _SPARSE
        mat_ptr[s + 1] = mat_ptr[s] + (local_dim * local_dim if dense_step else 0)
        sv_ptr[s + 1] = sv_ptr[s] + (0 if dense_step or sparse_step else local_dim)
        off_ptr[s + 1] = off_ptr[s] + local_dim
        comp_ptr[s + 1] = comp_ptr[s] + len(dims)
        sp_indptr_ptr[s + 1] = sp_indptr_ptr[s] + (
            len(sp_indptr_parts[-1]) if sparse_step else 0
        )
        sp_data_ptr[s + 1] = sp_data_ptr[s] + (
            len(sp_data_parts[-1]) if sparse_step else 0
        )

    def _join(parts: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    packed = (
        code,
        _join(mat_parts, np.complex128),
        mat_ptr,
        local_dims,
        _join(sv_column_parts, np.int64),
        _join(sv_value_parts, np.complex128),
        sv_ptr,
        _join(off_parts, np.int64),
        off_ptr,
        _join(comp_stride_parts, np.int64),
        _join(comp_dim_parts, np.int64),
        comp_ptr,
        num_cosets,
        _join(sp_indptr_parts, np.int64),
        _join(sp_index_parts, np.int64),
        _join(sp_data_parts, np.complex128),
        sp_indptr_ptr,
        sp_data_ptr,
    )
    return packed, scratch_rows
