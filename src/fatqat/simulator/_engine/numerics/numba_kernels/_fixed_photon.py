"""Numba numerics for states of one fixed total photon number.

The state, row order and gate blocks are those of
``numpy_kernels._fixed_photon``; these kernels walk the rows in order instead
of decoding chunks, so they need no scratch beyond a few occupations. A pass
splits the rows into ``chunks`` that run in parallel, each starting from its
first row's decoded occupation. Groups are disjoint and each is updated only
from its leader row, so chunks never write the same amplitude.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit, prange

from ....._backends.steps import ApplyMatrixStep
from ..numpy_kernels._fixed_photon import (
    _FixedPhotonState,
    _gate_blocks,
    _normalize,
)
from ._layouts import _MAX_THREADS, _MIN_SIZE_TO_THREAD, _choose_chunk_count


@njit(cache=True)
def _rank_span(
    counts: np.ndarray, first: int, span: np.ndarray, remaining: int
) -> int:  # pragma: no cover - compiled by Numba
    """Return the part of a row that columns ``first`` on contribute.

    As ``numpy_kernels._fixed_photon._rank_span``, for one occupation.
    """
    num_modes = counts.shape[0] - 1
    row = 0
    for j in range(span.shape[0]):
        within = num_modes - first - j
        row += counts[within, remaining] - counts[within, remaining - span[j]]
        remaining -= span[j]
    return row


@njit(cache=True)
def _unrank_one(
    counts: np.ndarray, row: int, photons: int, occupation: np.ndarray
) -> None:  # pragma: no cover - compiled by Numba
    """Write the public-order occupation of ``row``, as ``_unrank`` does."""
    num_modes = occupation.shape[0]
    remaining = photons
    for column in range(num_modes - 1):
        modes = num_modes - column
        level = 0
        while (
            level < remaining
            and counts[modes, remaining] - counts[modes, remaining - level - 1] <= row
        ):
            level += 1
        row -= counts[modes, remaining] - counts[modes, remaining - level]
        remaining -= level
        occupation[column] = level
    occupation[num_modes - 1] = remaining


@njit(cache=True)
def _next_occupation(occupation: np.ndarray) -> None:  # pragma: no cover - Numba
    """Advance a public-order occupation to the next row, in place.

    The next occupation in lexicographic order raises the last column that
    has photons after it by one and moves the rest of those photons, less
    one, to the final column.
    """
    last = occupation.shape[0] - 1
    tail = occupation[last]
    column = last - 1
    while column >= 0 and tail == 0:
        tail += occupation[column]
        occupation[column] = 0
        column -= 1
    if column < 0:
        return
    occupation[column] += 1
    occupation[last] = tail - 1


@njit(cache=True, parallel=True)
def _apply_blocks(
    amplitudes: np.ndarray,
    counts: np.ndarray,
    photons: int,
    columns: np.ndarray,
    first_column: int,
    last_column: int,
    offsets: np.ndarray,
    levels: np.ndarray,
    matrices: np.ndarray,
    chunks: int,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply each group's block in place, visiting each group from its leader.

    ``matrices[s]`` is the block of local photon number ``s``, zero-padded.
    A member differs from its leader only from ``first_column`` to
    ``last_column`` and holds the same photons there, so its row is the
    leader's plus the change in that span's part.
    """
    num_modes = counts.shape[0] - 1
    size = amplitudes.shape[0]
    width = last_column - first_column + 1
    per_chunk = (size + chunks - 1) // chunks
    for chunk in prange(chunks):  # pylint: disable=not-an-iterable
        start = chunk * per_chunk
        stop = min(size, start + per_chunk)
        if start >= stop:
            continue
        occupation = np.empty(num_modes, dtype=np.int64)
        _unrank_one(counts, start, photons, occupation)
        span = np.empty(width, dtype=np.int64)
        rows = np.empty(matrices.shape[1], dtype=np.int64)
        values = np.empty(matrices.shape[1], dtype=amplitudes.dtype)
        for row in range(start, stop):
            s = 0
            for j in range(columns.shape[0]):
                s += occupation[columns[j]]
            first = offsets[s]
            leader = True
            for j in range(columns.shape[0]):
                if occupation[columns[j]] != levels[first, j]:
                    leader = False
                    break
            if leader:
                remaining = photons
                for column in range(first_column):
                    remaining -= occupation[column]
                span[:] = occupation[first_column : last_column + 1]
                base = row - _rank_span(counts, first_column, span, remaining)
                group = offsets[s + 1] - first
                for a in range(group):
                    for j in range(columns.shape[0]):
                        span[columns[j] - first_column] = levels[first + a, j]
                    rows[a] = base + _rank_span(counts, first_column, span, remaining)
                    values[a] = amplitudes[rows[a]]
                for a in range(group):
                    total = 0j
                    for b in range(group):
                        total += matrices[s, a, b] * values[b]
                    amplitudes[rows[a]] = total
            _next_occupation(occupation)


@njit(cache=True, parallel=True)
def _project_rows(
    amplitudes: np.ndarray,
    counts: np.ndarray,
    photons: int,
    columns: np.ndarray,
    outcome: np.ndarray,
    chunks: int,
) -> None:  # pragma: no cover - compiled by Numba
    """Zero, in place, the rows that differ from ``outcome`` on ``columns``."""
    num_modes = counts.shape[0] - 1
    size = amplitudes.shape[0]
    per_chunk = (size + chunks - 1) // chunks
    for chunk in prange(chunks):  # pylint: disable=not-an-iterable
        start = chunk * per_chunk
        stop = min(size, start + per_chunk)
        if start >= stop:
            continue
        occupation = np.empty(num_modes, dtype=np.int64)
        _unrank_one(counts, start, photons, occupation)
        for row in range(start, stop):
            for j in range(columns.shape[0]):
                if occupation[columns[j]] != outcome[j]:
                    amplitudes[row] = 0.0
                    break
            _next_occupation(occupation)


@njit(cache=True)
def _levels_of_rows(
    counts: np.ndarray, photons: int, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Return the levels on ``columns`` of each row."""
    occupation = np.empty(counts.shape[0] - 1, dtype=np.int64)
    levels = np.empty((rows.shape[0], columns.shape[0]), dtype=np.int64)
    for i in range(rows.shape[0]):
        _unrank_one(counts, rows[i], photons, occupation)
        for j in range(columns.shape[0]):
            levels[i, j] = occupation[columns[j]]
    return levels


def _row_chunks(size: int, workers: int, force_parallel: bool) -> int:
    """Return how many parallel chunks a pass over ``size`` rows uses."""
    return _choose_chunk_count(
        size,
        size,
        _MIN_SIZE_TO_THREAD,
        max_workers=workers,
        force_parallel=force_parallel,
    )


def _apply_matrix_fixed_photon_numba(
    state: _FixedPhotonState,
    step: ApplyMatrixStep,
    dims: Sequence[int],
    *,
    workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> _FixedPhotonState:
    """Apply a photon-number-conserving local matrix block by block, in place."""
    sector = state.sector
    blocks = _gate_blocks(step.matrix, step.target_indices, dims, sector.photons)
    width = max(len(matrix) for matrix in blocks.matrices)
    matrices = np.zeros((len(blocks.matrices), width, width), dtype=complex)
    for s, matrix in enumerate(blocks.matrices):
        matrices[s, : len(matrix), : len(matrix)] = matrix
    _apply_blocks(
        state.amplitudes,
        sector.counts,
        sector.photons,
        blocks.columns,
        blocks.first,
        blocks.last,
        blocks.offsets,
        blocks.levels,
        matrices,
        _row_chunks(sector.size, workers, force_parallel),
    )
    return state


def _project_fixed_photon_numba(
    state: _FixedPhotonState,
    row: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
    *,
    workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> _FixedPhotonState:
    """Keep the rows that match ``row`` on the measured subsystems, in place."""
    del dims  # rows are not flat indices; the sector decodes them
    sector = state.sector
    columns = np.array(
        [sector.num_modes - 1 - q for q in measured_subsystems], dtype=np.int64
    )
    outcome = _levels_of_rows(
        sector.counts, sector.photons, np.array([row], dtype=np.int64), columns
    )[0]
    _project_rows(
        state.amplitudes,
        sector.counts,
        sector.photons,
        columns,
        outcome,
        _row_chunks(sector.size, workers, force_parallel),
    )
    _normalize(state.amplitudes)
    return state


def _fixed_photon_levels_numba(
    state: _FixedPhotonState, rows: np.ndarray, subsystems: Sequence[int]
) -> np.ndarray:
    """Return the measured subsystems' levels for sampled rows."""
    sector = state.sector
    columns = np.array([sector.num_modes - 1 - q for q in subsystems], dtype=np.int64)
    return _levels_of_rows(
        sector.counts,
        sector.photons,
        np.atleast_1d(np.asarray(rows, dtype=np.int64)),
        columns,
    )
