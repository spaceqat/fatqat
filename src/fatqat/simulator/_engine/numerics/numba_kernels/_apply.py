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
# - `_MIN_RUN_LENGTH`: shorter tiles leave too little work to vectorize.
# - `_MAX_RUN_DENSE_DIM`: a dense row costs ``D`` multiply-adds per amplitude
#   in either walk, so for larger ``D`` the arithmetic dominates and the extra
#   scratch traffic no longer pays off.
# - `_RUN_TILE`: runs are split into tiles of at most this many cosets.
# - `_RUN_ORDER_MIN_SIZE`: permutations are relabelled into address order
#   (`_run_walk_order`) only for states of at least this many amplitudes
#   (1 MiB). Larger states stream from memory, where in-order rows of adjacent
#   multi-qubit targets were up to 1.7x faster; at 16384 amplitudes and below
#   the order made no difference.
#
# The walk is chosen per range, since each thread allocates and touches its own
# scratch; the per-range cost of these checks is a few integer operations.
#
# - `_RUN_SCRATCH`: a tile's ``D`` scratch rows hold at most this many
#   amplitudes (128 KiB), so wider permutations use narrower tiles. A gate too
#   wide for a tile of `_MIN_RUN_LENGTH`, more than 1024 rows (over 10 qubits),
#   keeps the coset walk: its scratch would not fit, and short ranges of such
#   gates lost to the coset walk in review benchmarks. Diagonal gates have no
#   scratch and are not narrowed.
# - `_RUN_SMALL_SCRATCH`: a tile whose scratch exceeds this many amplitudes
#   (4 KiB) needs a range of at least two tiles to pay for it. One-tile ranges
#   with 32 KiB of scratch, such as a three-qubit gate on 14 qubits split over
#   eight threads, were slower than the coset walk in review benchmarks.
_MIN_RUN_LENGTH = 8
_MAX_RUN_DENSE_DIM = 6
_RUN_TILE = 256
_RUN_ORDER_MIN_SIZE = 1 << 16
_RUN_SCRATCH = 8192
_RUN_SMALL_SCRATCH = 256


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


@njit(cache=True, inline="always")
def _run_tile(
    scratch_rows, run_length
) -> tuple[int, int]:  # pragma: no cover - compiled by Numba
    """Return the run walk's tile width and the shortest range that may use it.

    ``scratch_rows`` is ``D``, or 0 for a diagonal gate. Both values are in
    cosets; a tile below `_MIN_RUN_LENGTH` means the run walk is not used.
    Kept branch-free: computing the tile under an ``if`` made LLVM
    compile the tile loops about twice as slow for runs of 8.
    """
    tile = min(run_length, _RUN_TILE, _RUN_SCRATCH // max(1, scratch_rows))
    small = scratch_rows * tile <= _RUN_SMALL_SCRATCH
    return tile, _MIN_RUN_LENGTH if small else 2 * tile


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
    tile,
    start,
    end,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply cosets ``[start, end)``, walking them in runs of ``run_length``.

    Coset ``k`` sits ``k % run_length`` positions after the start of run
    ``k // run_length``. The odometer only steps over the complement subsystems
    above the run. ``start`` and ``end`` may fall inside a run, so the result
    does not depend on how the cosets are split between threads. ``tile`` comes
    from `_run_tile` and is narrowed further to the range.
    """
    outer_strides = comp_strides[run_subsystems:]
    outer_dims = comp_dims[run_subsystems:]
    counter = np.empty(outer_strides.shape[0], dtype=np.int64)
    tile = min(tile, end - start)
    scratch_rows = 0 if code == _DIAGONAL else offsets.shape[0]
    scratch = np.empty((scratch_rows, tile), state.dtype)
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
                _apply_diagonal_tile(state, values, offsets, first, width)
            elif code == _PERMUTATION:
                _apply_permutation_tile(
                    state, columns, values, offsets, scratch, first, width
                )
            else:
                _apply_dense_tile(state, matrix, offsets, scratch, first, width)
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
    small dense gates whose runs allow a full tile, in a range long enough for
    it, use `_apply_contiguous_range` instead, which gives the same amplitudes.
    """
    if (
        code in (_DIAGONAL, _PERMUTATION)
        or code == _DENSE
        and offsets.shape[0] <= _MAX_RUN_DENSE_DIM
    ):
        run_subsystems, run_length = _contiguous_run(comp_strides, comp_dims)
        scratch_rows = 0 if code == _DIAGONAL else offsets.shape[0]
        tile, shortest = _run_tile(scratch_rows, run_length)
        if tile >= _MIN_RUN_LENGTH and end - start >= shortest:
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
                tile,
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


# --- run-walk preparation (host side, once per gate and layout) ---


def _run_walk_order(layout: tuple, cosets_per_range: int) -> np.ndarray | None:
    """Return the row order that puts a permutation's offsets in address order.

    Qubit targets give bit-reversed offsets (0, 128, 64, ...), which make a
    run-walk tile jump back and forth through memory. Returns ``None`` when the
    state is smaller than `_RUN_ORDER_MIN_SIZE`, when the offsets are already in
    order, or when ranges of ``cosets_per_range`` cannot take the run walk
    (`_apply_resolved_range` applies the same rule), so callers skip
    relabelling that cannot help. Gates are relabelled when they are prepared,
    never per application.
    """
    offsets, comp_strides, comp_dims, num_cosets = layout[:4]
    if offsets.shape[0] * num_cosets < _RUN_ORDER_MIN_SIZE:
        return None
    _, run_length = _contiguous_run(comp_strides, comp_dims)
    tile, shortest = _run_tile(offsets.shape[0], run_length)
    if tile < _MIN_RUN_LENGTH or cosets_per_range < shortest:
        return None
    if np.all(offsets[1:] > offsets[:-1]):
        return None
    return np.argsort(offsets)


def _reorder_permutation(
    order: np.ndarray, offsets: np.ndarray, columns: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relabel a permutation's rows in ``order``.

    Each output is still the same factor times the same input, so either walk
    gives unchanged results. Only permutations are relabelled: dense rows sum
    their columns in a fixed order, and the single-pass batch kernel writes
    diagonal values in matrix row order.
    """
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0])
    return offsets[order], rank[columns[order]], values[order]


# --- Python-facing matrix operations ---


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
