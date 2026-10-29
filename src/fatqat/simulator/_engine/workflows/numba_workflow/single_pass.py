"""Compiled single-pass evolution of a statevector parameter batch.

A noiseless plan evolves once and samples its terminal measurements from the
final state. For a parameter batch the plan is packed once by
`trajectories._compile_dynamic_plan`; the kernel fills each entry's
parametric gate matrices from its parameter values and evolves all entries in
parallel. Sampling stays with the Python single-pass workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import NamedTuple

import numpy as np
from numba import njit, prange

from ....._backends.steps import (
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    ResolvedStep,
)
from ...._execution_contract import _RunContext
from .....parameters import Parameter
from ...state import EvolutionState
from ...numerics.numba_kernels._apply import _DIAGONAL
from ...numerics.numba_kernels._parametric import _FORMULAS, _fill_parametric_matrix
from ..python_workflow import _PythonExecution
from .trajectories import (
    _NumbaTrajectoryExecution,
    _apply_matrix_step,
    _compile_dynamic_plan,
)


class _SVBatch(NamedTuple):
    """A packed plan plus the parametric gates the kernel fills per entry.

    Per parametric gate: its gate-table index, formula, and for each formula
    field either a column of ``values`` or, at ``-1``, a constant. An ordinary
    plan has no parametric gates and one entry.
    """

    packed: _NumbaTrajectoryExecution
    gate: np.ndarray
    formula: np.ndarray
    column: np.ndarray  # (gates, 3) value-table column per field, -1 if constant
    constant: np.ndarray  # (gates, 3) value of each constant field
    levels: np.ndarray  # (gates, 2) subspace levels, (0, 1) otherwise
    values: np.ndarray  # (entries, columns) parameter values, float64


class _NumbaSinglePassExecution(NamedTuple):
    """Packed evolution plus the Python payload that samples evolved states."""

    batch: _SVBatch
    sampling: _PythonExecution


def _placeholder_plan(
    plan: Sequence[ResolvedStep], dims: tuple[int, ...]
) -> tuple[ResolvedStep, ...]:
    """Stand in for each parametric gate with an identity matrix step.

    The placeholder keeps the gate's targets, condition, and kernel key, so
    planning and packing read it as the bound gate.
    """
    return tuple(
        (
            ApplyMatrixStep(
                matrix=np.eye(
                    prod(dims[i] for i in step.target_indices), dtype=complex
                ),
                target_indices=step.target_indices,
                condition=step.condition,
                kernel_key=step.implementation._kernel_key(
                    step.operation, targets=step.targets
                ),
            )
            if isinstance(step, ApplyParametricMatrixStep)
            else step
        )
        for step in plan
    )


def _pack_sv_batch(
    plan: Sequence[ResolvedStep],
    template: Sequence[ResolvedStep],
    dims: tuple[int, ...],
    build_apply_layout: Callable[[tuple[int, ...]], tuple],
    resolve_structure: Callable[[ApplyMatrixStep], tuple],
) -> _SVBatch:
    """Pack ``template``, ``plan``'s placeholder plan, once for the kernel.

    The kernel overwrites each placeholder per entry from the injected value
    columns of ``plan``'s parametric steps.
    """
    rows: list[tuple] = []
    columns: dict[int, int] = {}
    tables: list[np.ndarray] = []
    gate = 0
    for step, packed_step in zip(plan, template, strict=True):
        if isinstance(step, ApplyParametricMatrixStep):
            formula, fields = _FORMULAS[packed_step.kernel_key]
            given = dict(step.values)
            column, constant = [-1, -1, -1], [0.0, 0.0, 0.0]
            for index, name in enumerate(fields):
                value = getattr(step.operation, name)
                if isinstance(value, Parameter):
                    if id(given[name]) not in columns:
                        columns[id(given[name])] = len(tables)
                        tables.append(given[name])
                    column[index] = columns[id(given[name])]
                else:
                    constant[index] = float(value)
            levels = getattr(step.operation, "subspace", (0, 1))
            rows.append((gate, formula, column, constant, levels))
        if isinstance(packed_step, ApplyMatrixStep):
            gate += 1
    return _SVBatch(
        packed=_compile_dynamic_plan(
            list(template), dims, build_apply_layout, resolve_structure
        ),
        gate=np.array([row[0] for row in rows], dtype=np.int64),
        formula=np.array([row[1] for row in rows], dtype=np.int64),
        column=np.array([row[2] for row in rows], dtype=np.int64).reshape(-1, 3),
        constant=np.array([row[3] for row in rows], dtype=np.float64).reshape(-1, 3),
        levels=np.array([row[4] for row in rows], dtype=np.int64).reshape(-1, 2),
        values=(
            np.column_stack([np.asarray(t, dtype=np.float64) for t in tables])
            if tables
            else np.empty((1, 0), dtype=np.float64)
        ),
    )


def _evolve_sv_batch(
    batch: _SVBatch,
    entries: slice,
    context: _RunContext,
    *,
    initial_state: EvolutionState[np.ndarray],
) -> np.ndarray:
    """Return the final statevector of each entry in ``entries``, one per row.

    Measurement steps are skipped: a single-pass plan defers them to sampling.
    """
    arrays = batch.packed.plan_arrays
    state_size = prod(context.system_dims) if context.system_dims else 1
    custom_start = initial_state.quantum is not None
    start = (
        np.ascontiguousarray(initial_state.quantum, dtype=np.complex128).reshape(
            state_size
        )
        if custom_start
        else np.empty(0, dtype=np.complex128)
    )
    # Positions follow `_compile_dynamic_plan`'s ``plan_arrays`` order: the step
    # sequencer (0-1) and the gate table with its backing pools (6-17).
    return _run_sv_batch(
        *arrays[0:2],
        *arrays[6:18],
        batch.gate,
        batch.formula,
        batch.column,
        batch.constant,
        batch.levels,
        batch.values[entries],
        start,
        state_size,
        custom_start,
    )


@njit(cache=True, parallel=True)
def _run_sv_batch(
    step_kind,  # 0=gate; every other kind is a deferred measurement here
    step_data,  # gate index into the gate table
    ap_mat_ptr,  # gate table, as packed by `_compile_dynamic_plan`
    ap_dim,
    ap_off_ptr,
    ap_comp_ptr,
    ap_comp_len,
    ap_code,
    mat_flat,  # parametric slots hold placeholders until filled per entry
    off_flat,
    col_flat,
    val_flat,
    comp_stride_flat,
    comp_dim_flat,
    pg_gate,  # parametric gates: gate index, formula, and field sources
    pg_formula,
    pg_column,
    pg_constant,
    pg_levels,
    values,  # (entries, columns) parameter values of these entries
    start,  # custom-state template; empty when the default state is requested
    state_size,
    custom_start,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Bind each entry's parametric gates, then evolve it; entries run in parallel.

    Each entry fills its own copy of the matrix pool, plus the diagonal values
    of gates declared diagonal for every parameter value, before applying the
    packed gate sequence.
    """
    n_entries = values.shape[0]
    states = np.empty((n_entries, state_size), dtype=np.complex128)
    for entry in prange(n_entries):  # pylint: disable=not-an-iterable
        matrices = mat_flat.copy()
        diagonals = val_flat.copy()
        args = np.empty(3, dtype=np.float64)
        for p in range(pg_gate.shape[0]):
            gate = pg_gate[p]
            for f in range(3):
                column = pg_column[p, f]
                args[f] = values[entry, column] if column >= 0 else pg_constant[p, f]
            d = ap_dim[gate]
            first = ap_mat_ptr[gate]
            matrix = matrices[first : first + d * d].reshape(d, d)
            _fill_parametric_matrix(
                pg_formula[p],
                args[0],
                args[1],
                args[2],
                pg_levels[p, 0],
                pg_levels[p, 1],
                matrix,
            )
            if ap_code[gate] == _DIAGONAL:
                offset = ap_off_ptr[gate]
                for r in range(d):
                    diagonals[offset + r] = matrix[r, r]
        if custom_start:
            state = start.copy()
        else:
            state = np.zeros(state_size, dtype=np.complex128)
            state[0] = 1.0 + 0.0j
        for st in range(step_kind.shape[0]):
            if step_kind[st] == 0:
                _apply_matrix_step(
                    state,
                    step_data[st],
                    ap_mat_ptr,
                    ap_dim,
                    ap_off_ptr,
                    ap_comp_ptr,
                    ap_comp_len,
                    ap_code,
                    matrices,
                    off_flat,
                    col_flat,
                    diagonals,
                    comp_stride_flat,
                    comp_dim_flat,
                    state_size,
                )
        states[entry] = state
    return states
