"""Shared coset arithmetic for Numba matrix engines."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Gate-matrix structure codes. A diagonal or (phased) permutation matrix has at
# most one nonzero per row, so its application is a single multiply per amplitude
# instead of the D-term dot product a dense row needs - a machine-independent
# constant-factor win that covers most common gates (Z/S/T/RZ/CPhase/CZ are
# diagonal; X/CX/CCX/SWAP/iSwap are permutations).
_DENSE = 0
_DIAGONAL = 1
_PERMUTATION = 2
# CSR is selected during operand preparation, not by `_classify_matrix`.
_SPARSE = 3


@njit(cache=True)
def _classify_matrix(
    matrix: np.ndarray, columns: np.ndarray, values: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Classify ``matrix`` and fill the single-nonzero-per-row description.

    For each row ``r`` records the column ``columns[r]`` of its (sole) nonzero and
    the value ``values[r]`` there. Returns ``_DIAGONAL`` if every nonzero is on
    the diagonal, ``_PERMUTATION`` if each row has exactly one nonzero (off the
    diagonal somewhere), else ``_DENSE`` (``columns`` / ``values`` unused). Gate
    matrices are analytic, so structural zeros are exact - no tolerance needed.
    """
    d = matrix.shape[0]
    is_diagonal = True
    for r in range(d):
        count = 0
        column = 0
        for c in range(d):
            if matrix[r, c].real != 0.0 or matrix[r, c].imag != 0.0:
                count += 1
                column = c
        if count != 1:
            return _DENSE
        columns[r] = column
        values[r] = matrix[r, column]
        if column != r:
            is_diagonal = False
    return _DIAGONAL if is_diagonal else _PERMUTATION


@njit(cache=True)
def _spread_base(
    start: int, comp_strides: np.ndarray, comp_dims: np.ndarray, counter: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Spread coset index ``start`` into odometer ``counter``; return its flat base."""
    base = 0
    remainder = start
    for i in range(comp_strides.shape[0]):
        digit = remainder % comp_dims[i]
        remainder = remainder // comp_dims[i]
        counter[i] = digit
        base += digit * comp_strides[i]
    return base


@njit(cache=True, inline="always")
def _advance_base(
    base: int, comp_strides: np.ndarray, comp_dims: np.ndarray, counter: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Advance the odometer one coset, returning the next flat base."""
    for i in range(comp_strides.shape[0]):
        counter[i] += 1
        base += comp_strides[i]
        if counter[i] < comp_dims[i]:
            return base
        counter[i] = 0
        base -= comp_strides[i] * comp_dims[i]
    return base


@njit(cache=True)
def _apply_dense_range(
    state, matrix, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Dense ``D x D`` application to cosets ``[start, end)``: ``D``-term dot per row.

    Cosets partition the amplitudes by non-target ("complement") digits; within a
    coset the ``D`` target amplitudes sit at ``base + offsets[c]`` and the matrix
    maps them among themselves. ``base`` is spread from ``start`` once, then the
    range is walked with a division-free odometer; disjoint cosets keep the result
    independent of how the range is split.
    """
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    updated = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for r in range(local_dim):
            acc = 0.0 + 0.0j
            for c in range(local_dim):
                acc += matrix[r, c] * state[base + offsets[c]]
            updated[r] = acc
        for r in range(local_dim):
            state[base + offsets[r]] = updated[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_diagonal_range(
    state, diagonal, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Diagonal application: scale each amplitude in place (no gather, no dot)."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    for _ in range(start, end):
        for r in range(local_dim):
            state[base + offsets[r]] *= diagonal[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_permutation_range(
    state, columns, values, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Permutation application: one gather then one scaled scatter per coset."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    gathered = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for c in range(local_dim):
            gathered[c] = state[base + offsets[c]]
        for r in range(local_dim):
            state[base + offsets[r]] = values[r] * gathered[columns[r]]
        base = _advance_base(base, comp_strides, comp_dims, counter)


# --- contiguous-run walk (diagonal, permutation, and small dense gates) ---
#
# Subsystem 0 varies fastest in the flat layout. If the gate does not act on
# any subsystem below its lowest target, consecutive cosets sit at consecutive
# flat positions. A run of such cosets can then be applied as plain array loops:
# the odometer advances once per run instead of once per coset, and LLVM
# vectorizes the inner loops.
#
# - `_MIN_RUN_LENGTH`: shorter runs leave too little work to vectorize.
# - `_MAX_RUN_DENSE_DIM`: a dense row costs ``D`` multiply-adds per amplitude
#   in either walk, so for larger ``D`` the arithmetic dominates and the extra
#   scratch traffic no longer pays off.
# - `_RUN_TILE`: runs are split into tiles of at most this many cosets.
# - `_RUN_SCRATCH`: a tile's ``D`` scratch rows hold at most this many
#   amplitudes, so wide permutations use narrower tiles. A tile never drops
#   below `_MIN_RUN_LENGTH` cosets, where per-row overhead would dominate.
#
# A tile also never exceeds the cosets in the range. Each thread allocates and
# touches its own scratch, so sizing it for a full tile when a thread's chunk
# holds only a few cosets costs more than the chunk's work.
_MIN_RUN_LENGTH = 8
_MAX_RUN_DENSE_DIM = 6
_RUN_TILE = 256
_RUN_SCRATCH = 8192


@njit(cache=True)
def _contiguous_run(
    comp_strides: np.ndarray, comp_dims: np.ndarray
) -> tuple[int, int]:  # pragma: no cover - compiled by Numba
    """Return how many complement subsystems form the run, and its length.

    Complement subsystems are ordered by stride. The leading ones lie below
    every target exactly as long as each stride equals the product of the
    dimensions before it; their cosets are then consecutive flat positions.
    """
    run = 1
    count = 0
    while count < comp_strides.shape[0] and comp_strides[count] == run:
        run *= comp_dims[count]
        count += 1
    return count, run


@njit(cache=True)
def _rows_in_address_order(
    offsets, columns, values
):  # pragma: no cover - compiled by Numba
    """Relabel a diagonal or permutation gate so its rows run in address order.

    Qubit targets give bit-reversed offsets (0, 128, 64, ...), which make a
    tile jump back and forth through memory. Each output is still the same
    factor times the same input, so results are unchanged. Relabelling once
    keeps the tile loops plain ``range`` loops; iterating an index array
    inside them was measurably slower for small gates.
    """
    local_dim = offsets.shape[0]
    order = np.arange(local_dim)
    for r in range(1, local_dim):
        if offsets[r] < offsets[r - 1]:
            order = np.argsort(offsets)
            break
    rank = np.empty(local_dim, np.int64)
    for i in range(local_dim):
        rank[order[i]] = i
    row_columns = np.empty(local_dim, np.int64)
    for i in range(local_dim):
        row_columns[i] = rank[columns[order[i]]]
    return offsets[order], row_columns, values[order]


@njit(cache=True, inline="always")
def _apply_dense_tile(
    state, matrix, offsets, scratch, first, width
) -> None:  # pragma: no cover - compiled by Numba
    """Apply a dense matrix to ``width`` consecutive cosets starting at ``first``.

    Each output row is summed from zero in column order, like
    `_apply_dense_range`, so both walks give bit-identical results.
    """
    local_dim = offsets.shape[0]
    for r in range(local_dim):
        row = scratch[r]
        coefficient = matrix[r, 0]
        source = first + offsets[0]
        for j in range(width):
            row[j] = (0.0 + 0.0j) + coefficient * state[source + j]
        for c in range(1, local_dim):
            coefficient = matrix[r, c]
            source = first + offsets[c]
            for j in range(width):
                row[j] += coefficient * state[source + j]
    for r in range(local_dim):
        row = scratch[r]
        target = first + offsets[r]
        for j in range(width):
            state[target + j] = row[j]


@njit(cache=True, inline="always")
def _apply_diagonal_tile(
    state, diagonal, offsets, first, width
) -> None:  # pragma: no cover - compiled by Numba
    """Apply a diagonal matrix to ``width`` consecutive cosets."""
    for r in range(offsets.shape[0]):
        factor = diagonal[r]
        target = first + offsets[r]
        for j in range(width):
            state[target + j] *= factor


@njit(cache=True, inline="always")
def _apply_permutation_tile(
    state, columns, values, offsets, scratch, first, width
) -> None:  # pragma: no cover - compiled by Numba
    """Apply a phased permutation to ``width`` consecutive cosets.

    The inputs are copied to ``scratch`` first, since outputs overwrite them.
    """
    local_dim = offsets.shape[0]
    for c in range(local_dim):
        row = scratch[c]
        source = first + offsets[c]
        for j in range(width):
            row[j] = state[source + j]
    for r in range(local_dim):
        row = scratch[columns[r]]
        factor = values[r]
        target = first + offsets[r]
        for j in range(width):
            state[target + j] = factor * row[j]


@njit(cache=True)
def _apply_contiguous_range(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    run_subsystems,
    run_length,
    start,
    end,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply cosets ``[start, end)``, walking them in runs of ``run_length``.

    Coset ``k`` sits ``k % run_length`` positions after the start of run
    ``k // run_length``. The odometer only steps over the complement subsystems
    above the run. ``start`` and ``end`` may fall inside a run, so the result
    does not depend on how the cosets are split between threads.
    """
    outer_strides = comp_strides[run_subsystems:]
    outer_dims = comp_dims[run_subsystems:]
    counter = np.empty(outer_strides.shape[0], dtype=np.int64)
    local_dim = offsets.shape[0]
    tile = min(
        run_length,
        _RUN_TILE,
        end - start,
        max(_MIN_RUN_LENGTH, _RUN_SCRATCH // local_dim),
    )
    scratch = np.empty((0 if code == _DIAGONAL else local_dim, tile), state.dtype)
    row_offsets, row_columns, row_values = offsets, columns, values
    if code != _DENSE:  # dense rows sum their columns in a fixed order
        row_offsets, row_columns, row_values = _rows_in_address_order(
            offsets, columns, values
        )
    outer = start // run_length
    position = start - outer * run_length
    base = _spread_base(outer, outer_strides, outer_dims, counter)
    remaining = end - start
    while remaining > 0:
        stop = min(run_length, position + remaining)
        while position < stop:
            width = min(tile, stop - position)
            first = base + position
            if code == _DIAGONAL:
                _apply_diagonal_tile(state, row_values, row_offsets, first, width)
            elif code == _PERMUTATION:
                _apply_permutation_tile(
                    state,
                    row_columns,
                    row_values,
                    row_offsets,
                    scratch,
                    first,
                    width,
                )
            else:
                _apply_dense_tile(state, matrix, row_offsets, scratch, first, width)
            position += width
            remaining -= width
        if remaining > 0:
            base = _advance_base(base, outer_strides, outer_dims, counter)
            position = 0


# --- coset-range dispatch ---


@njit(cache=True, inline="always")
def _apply_resolved_range(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    start,
    end,
    indptr=None,
) -> None:  # pragma: no cover - compiled by Numba
    """Route one coset range to the structure-specialized kernel.

    Dense uses the full ``matrix``; diagonal uses ``values[row]`` only.
    Permutation uses one source ``columns[row]`` and multiplier ``values[row]``
    per output row. CSR uses ``columns`` and ``values`` for its nonzero entries,
    with ``indptr`` delimiting rows. Non-CSR representations omit ``indptr``;
    each kernel ignores operands it does not use. Diagonal, permutation, and
    small dense gates whose cosets form long enough runs use
    `_apply_contiguous_range` instead, which gives the same amplitudes.
    """
    if (
        code in (_DIAGONAL, _PERMUTATION)
        or code == _DENSE
        and offsets.shape[0] <= _MAX_RUN_DENSE_DIM
    ):
        run_subsystems, run_length = _contiguous_run(comp_strides, comp_dims)
        if run_length >= _MIN_RUN_LENGTH:
            _apply_contiguous_range(
                state,
                code,
                matrix,
                columns,
                values,
                offsets,
                comp_strides,
                comp_dims,
                run_subsystems,
                run_length,
                start,
                end,
            )
            return
    if code == _SPARSE and indptr is not None:
        _apply_sparse_range(
            state, indptr, columns, values, offsets, comp_strides, comp_dims, start, end
        )
    elif code == _DIAGONAL:
        _apply_diagonal_range(
            state, values, offsets, comp_strides, comp_dims, start, end
        )
    elif code == _PERMUTATION:
        _apply_permutation_range(
            state, columns, values, offsets, comp_strides, comp_dims, start, end
        )
    else:
        _apply_dense_range(state, matrix, offsets, comp_strides, comp_dims, start, end)


@njit(cache=True)
def _apply_resolved_serial(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    indptr=None,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Single-threaded application of an already-classified gate."""
    _apply_resolved_range(
        state,
        code,
        matrix,
        columns,
        values,
        offsets,
        comp_strides,
        comp_dims,
        0,
        num_cosets,
        indptr,
    )
    return state


@njit(cache=True, parallel=True)
def _apply_resolved_parallel(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    n_chunks,
    indptr=None,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Parallel application of an already-classified gate over coset chunks."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        start = chunk * num_cosets // n_chunks
        end = (chunk + 1) * num_cosets // n_chunks
        if start < end:
            _apply_resolved_range(
                state,
                code,
                matrix,
                columns,
                values,
                offsets,
                comp_strides,
                comp_dims,
                start,
                end,
                indptr,
            )
    return state


# --- sparse coset kernel (density-matrix channel super-operators) ---
#
# A channel super-operator ``sum_i kron(K_i, conj(K_i))`` is often mostly zero
# even when neither diagonal nor a permutation (two-qubit depolarizing fills
# ``2 d**2 - d`` of ``d**4`` entries), which `_classify_matrix` calls dense.
# These kernels walk a CSR of the exact nonzeros in column order - bit-identical
# to the dense walk, at a fraction of the multiplies.


@njit(cache=True)
def _apply_sparse_range(
    state, indptr, indices, data, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Sparse ``D x D`` (CSR) application to cosets ``[start, end)``."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    gathered = np.empty(local_dim, dtype=np.complex128)
    updated = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for c in range(local_dim):
            gathered[c] = state[base + offsets[c]]
        for r in range(local_dim):
            acc = 0.0 + 0.0j
            for k in range(indptr[r], indptr[r + 1]):
                acc += data[k] * gathered[indices[k]]
            updated[r] = acc
        for r in range(local_dim):
            state[base + offsets[r]] = updated[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


def _apply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    layout: tuple,
    code: int,
    columns: np.ndarray,
    values: np.ndarray,
    *,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply a classified matrix to native storage using a precomputed layout.

    The layout describes flat storage, including column-batched operators and
    vectorized density matrices. Buffer conversion and flattening belong here;
    callers supply numerical operands without managing the kernel's flat view.
    Contiguous complex128 input is updated in place, with its shape preserved.
    CSR operands select sparse execution; otherwise ``code`` selects dense,
    diagonal, or permutation arithmetic. The layout controls kernel threading.
    """
    state = np.ascontiguousarray(state, dtype=np.complex128)
    flat = state if state.ndim == 1 else state.reshape(-1)
    offsets, comp_strides, comp_dims, num_cosets, n_chunks = layout
    indptr = None
    if sparse is not None:
        code = _SPARSE
        indptr, columns, values = sparse
    else:
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    if n_chunks <= 1:
        result = _apply_resolved_serial(
            flat,
            code,
            matrix,
            columns,
            values,
            offsets,
            comp_strides,
            comp_dims,
            num_cosets,
            indptr,
        )
    else:
        result = _apply_resolved_parallel(
            flat,
            code,
            matrix,
            columns,
            values,
            offsets,
            comp_strides,
            comp_dims,
            num_cosets,
            n_chunks,
            indptr,
        )
    return result if state.ndim == 1 else result.reshape(state.shape)
