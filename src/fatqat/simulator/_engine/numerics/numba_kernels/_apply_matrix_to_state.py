"""Python-callable Numba matrix operations over explicit native state.

Engines own cache lifetimes and execution policy. These functions use those
numerical resources without retaining an engine or an evolving state. Target
layouts are complete before execution and are only read by these operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ....._backends.steps import ApplyChannelStep, ApplyMatrixStep
from ._apply import _apply_matrix, _classify_matrix

if TYPE_CHECKING:
    from ._matrix_cache import _MatrixStructureCache, _SuperoperatorCache


def _apply_matrix_statevector(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
    *,
    structures: _MatrixStructureCache,
    layouts: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Apply one gate using cached classification and target layout."""
    layout = layouts[tuple(step.target_indices)]
    code, layout, columns, values = structures.prepare(step, layout)
    return _apply_matrix(state, step.matrix, layout, code, columns, values)


def _apply_superoperator_to_state(
    state: np.ndarray,
    step: ApplyMatrixStep | ApplyChannelStep,
    dims: Sequence[int],
    *,
    superoperators: _SuperoperatorCache,
    layouts: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Apply a cached local superoperator without copying contiguous state.

    Gates and exact Kraus sums share this numerical operation. A sparse local
    superoperator uses CSR arithmetic; the evolving state remains dense.
    """
    layout = layouts[tuple(step.target_indices)]
    superop, code, layout, columns, values, sparse = superoperators.prepare(
        step, layout
    )
    return _apply_matrix(state, superop, layout, code, columns, values, sparse=sparse)


def _apply_unclassified_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    *,
    layouts: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Classify a bare Kraus matrix once and use the shared matrix launcher."""
    matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    columns = np.empty(matrix.shape[0], dtype=np.int64)
    values = np.empty(matrix.shape[0], dtype=np.complex128)
    code = int(_classify_matrix(matrix, columns, values))
    targets = tuple(targets)
    layout = layouts[targets]
    return _apply_matrix(state, matrix, layout, code, columns, values)
