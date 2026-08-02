"""Numba matrix-family engines.

`NumbaSVEngine` (statevector) and `NumbaDMEngine` (density matrix) reuse
every semantics-agnostic piece of their NumPy twins - strategy selection, the
fast and per-shot paths, ``initialize`` / ``measure_subsystems`` dispatch - and
replace only the numeric kernels with Numba-jitted loops. Both are reachable
via ``Simulator(method=..., runtime="numba")``.

Kernel selection is key-driven: an `ApplyMatrixStep` carries the canonical
identity of the implementation that produced its matrix (``kernel_key``), and
each engine resolves which specialized kernel a step gets once per plan - by
declared identity or one `_classify_matrix` content scan, never per application
or per shot. `_KERNEL_SPECS` is the statevector key-to-structure table; the
density matrix derives its table from the same declarations because the
Kronecker product preserves structure (see below).

- `NumbaSVEngine` replaces gate application (`apply` / `_apply_local`),
  `probabilities`, `sample_indices`, and `collapse`. ``measure_subsystems`` /
  ``reset_subsystems`` delegate to those, so they are Numba-routed without an
  override.
- `NumbaDMEngine` replaces `apply` / `apply_channel`, `probabilities`,
  `sample_indices`, and `collapse`. It reuses the *same* coset-walk kernels as
  the statevector path by viewing ``rho`` (shape ``(size, size)``) as a flat
  vector over a doubled ``2n``-subsystem system: ``n`` bra subsystems (strides
  ``prod(dims[:q])``) then ``n`` ket subsystems (strides ``size *
  prod(dims[:q])``). The sandwich ``M rho M^dagger`` is the single
  super-operator ``kron(M, conj(M))`` on ``vec(rho)``, and a channel is
  ``sum_i kron(K_i, conj(K_i))`` - each one coset walk over the combined
  ket+bra super-target. The Kronecker product preserves structure, so a gate's
  declared identity transfers to its super-operator: a declared-dense key skips
  the scan, everything else takes one scan of the super-operator, cached per
  step. ``4^n`` is memory-bound and each gate is one pass, so DM parallelizes
  later than the statevector path (`_MIN_SIZE_TO_THREAD_DM`). Reset stays the
  inherited NumPy partial-trace channel.

Noise math lives in ``noise.nb`` (quantum-jump branch selection for the
statevector, the channel super-operator for the density matrix, readout-error
resampling, and the fused-kernel plan flattening); *applying* a Kraus operator
is a plain local-matrix application on the coset kernels here. This module
imports ``noise.nb``; the reverse is forced never to happen (see its docstring).

The fused statevector channel path weighs branches the way Aer's trajectory
sampler does, not the way the NumPy reference does: instead of materializing
every ``K_i|psi>`` and norming it, it forms the targets' reduced density matrix
``rho_T`` once (`_reduced_density`) and reads each branch probability off it as
``Tr(K_i^dagger K_i rho_T)``, then applies only the chosen operator (see
`_channel_step`). A mathematically equal but numerically distinct estimator, so
it stays seed-reproducible and agrees with NumPy in distribution, not per-seed
bit-for-bit.

Parallelism has two independent axes: ``EngineConfig``'s ``max_workers`` /
``parallel_mode`` distribute dynamic shots across OS processes (reaching only
the inherited NumPy per-shot path), while ``numba_parallel`` switches this
module's in-process thread parallelism for a whole run (`_thread_scope`).

The RNG draw stays in NumPy - a ``np.random.Generator`` cannot cross into
nopython code - so uniforms are drawn with ``rng`` and the inverse-CDF search
runs in Numba. This consumes the stream identically to ``rng.choice``, so
counts stay reproducible per engine (float-summation differences from NumPy
mean they are not bit-identical across engines - the documented contract,
see `np.py`).

Numba compiles kernels lazily on first call and is an optional dependency (the
``numba`` group), so this module is never imported from ``fatqat.simulator._engine``'s
``__init__``; import it explicitly.

Conventions match `np.py`: little-endian flat indexing (subsystem ``q`` has
place value ``prod(dims[:q])``, subsystem 0 least-significant) and a local gate
matrix whose most-significant index digit is ``targets[0]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from math import prod, sqrt

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from ..._backends.engine_contract import _EngineConfig as EngineConfig, RawResult
from ..._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    BuiltinKernelKey,
    MeasurementStep,
    ResetStep,
)
from ...noise.nb import (
    _compile_channel_table,
    _compile_readout_table,
    _inverse_cdf_pick,
    _jump_branch_kernel,
    _kraus_stack,
    _kraus_superop_kernel,
    _report_digit_kernel,
)
from ...result import reduce_to_counts
from .np import NumpyDMEngine, NumpySVEngine

_MAX_THREADS = get_num_threads()
# A coset walk goes parallel only once each worker thread would get at least
# this many amplitudes of work; below that the parallel-region launch/sync cost
# outweighs the memory-bandwidth-bound work it saves. Expressed per-thread so
# the absolute floor scales with the core count (the dominant machine variable)
# as ``_MAX_THREADS * grain``, not a fixed size. The grains reproduce the
# previously hand-tuned 2^15 / 2^18 floors at ~16 threads; the density matrix's
# 4^n pass is more bandwidth-bound, so it takes a larger grain. The floor
# affects speed only - chunking is bit-identical - so a coarse grain suffices.
_GRAIN_TO_THREAD = 1 << 11  # statevector: min amplitudes per thread
_GRAIN_TO_THREAD_DM = 1 << 14  # density matrix: min amplitudes per thread
_MIN_SIZE_TO_THREAD = _MAX_THREADS * _GRAIN_TO_THREAD
_MIN_SIZE_TO_THREAD_DM = _MAX_THREADS * _GRAIN_TO_THREAD_DM


@contextmanager
def _thread_scope(config: EngineConfig):
    """Confine a run's Numba parallelism to one thread when asked to.

    ``numba_parallel=False`` means "this run must not claim Numba worker
    threads" - the knob a caller needs when *they* are the one parallelizing
    (several independent circuits at once) and a per-run thread pool would
    oversubscribe the machine. Every kernel here is compiled once, with
    ``parallel=True`` where it uses `prange`, so parallelism cannot be
    recompiled away per call; `set_num_threads(1)` collapses those same
    `prange` loops onto the calling thread instead, which is exactly "off".

    Numba's thread count is process-wide, so it is set immediately around the
    run and restored afterwards. Two concurrent runs in one process are already
    outside the engine's contract (a backend instance is single-threaded
    use only), and a worker process spawned by the NumPy per-shot path gets its
    own pool this cannot reach.
    """
    if config.numba_parallel:
        yield
        return
    previous = get_num_threads()
    set_num_threads(1)
    try:
        yield
    finally:
        set_num_threads(previous)


def _plan_chunks(
    num_cosets: int, size: int, min_parallel_size: int = _MIN_SIZE_TO_THREAD
) -> int:
    """Parallel chunk count for a gate on a ``size``-amplitude state.

    Returns 1 (the serial kernel, no parallel-region overhead) for states below
    ``min_parallel_size``; otherwise splits the cosets across all worker threads.
    """
    if size < min_parallel_size:
        return 1
    return max(1, min(_MAX_THREADS, num_cosets))


def _compute_apply_plan(
    dims: Sequence[int],
    targets: Sequence[int],
    min_parallel_size: int = _MIN_SIZE_TO_THREAD,
) -> tuple:
    """Precompute the strided-block kernel layout for a target tuple.

    Returns ``offsets[c]`` (the flat offset of local index ``c``, with
    ``targets[0]`` most-significant), the flat strides/dimensions of the
    complement (non-target) subsystems the kernel odometers over, the coset
    count, and the parallel chunk count for this many cosets.

    Depends only on ``dims`` and ``targets``, so the statevector path calls it
    with the physical system dims while the density-matrix path calls it with
    the doubled ``bra + ket`` dims (see the module docstring).
    """
    n = len(dims)
    local_dims = [dims[t] for t in targets]
    local_dim = prod(local_dims)

    strides = [prod(dims[:q]) for q in range(n)]
    target_strides = [strides[t] for t in targets]
    # Mixed-radix place values with targets[0] most-significant.
    local_places = [1] * len(targets)
    for j in range(len(targets) - 2, -1, -1):
        local_places[j] = local_places[j + 1] * local_dims[j + 1]

    offsets = np.empty(local_dim, dtype=np.int64)
    for c in range(local_dim):
        offset = 0
        for j, stride in enumerate(target_strides):
            offset += ((c // local_places[j]) % local_dims[j]) * stride
        offsets[c] = offset

    target_set = set(targets)
    complement = [q for q in range(n) if q not in target_set]
    comp_strides = np.array([strides[q] for q in complement], dtype=np.int64)
    comp_dims = np.array([dims[q] for q in complement], dtype=np.int64)
    size = prod(dims)
    num_cosets = size // local_dim
    n_chunks = _plan_chunks(num_cosets, size, min_parallel_size)
    return offsets, comp_strides, comp_dims, num_cosets, n_chunks


def _measured_layout(
    dims: Sequence[int], subsystems: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flat strides and dimensions of the measured subsystems for the kernels."""
    strides = np.array([prod(dims[:q]) for q in subsystems], dtype=np.int64)
    measured_dims = np.array([dims[q] for q in subsystems], dtype=np.int64)
    return strides, measured_dims


def _run_resolved(
    state: np.ndarray,
    matrix: np.ndarray,
    plan: tuple,
    code: int,
    columns: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Launch an already-classified matrix through a precomputed apply plan.

    Routes to the serial or parallel coset kernel by the plan's chunk count.
    Shared by the statevector launch path and the density-matrix
    super-operator pass; ``state`` must be contiguous ``complex128`` and is
    updated in place (and returned).
    """
    offsets, comp_strides, comp_dims, num_cosets, n_chunks = plan
    if n_chunks <= 1:
        return _apply_resolved_serial(
            state,
            code,
            matrix,
            columns,
            values,
            offsets,
            comp_strides,
            comp_dims,
            num_cosets,
        )
    return _apply_resolved_parallel(
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
    )


def _run_resolved_sparse(
    state: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    plan: tuple,
) -> np.ndarray:
    """Launch a CSR super-operator through a precomputed apply plan.

    The sparse analog of `_run_resolved`: same serial/parallel routing by the
    plan's chunk count, same in-place update, but the coset kernel walks the
    CSR nonzeros instead of a dense matrix (see the sparse kernels above).
    """
    offsets, comp_strides, comp_dims, num_cosets, n_chunks = plan
    if n_chunks <= 1:
        return _apply_sparse_serial(
            state, indptr, indices, data, offsets, comp_strides, comp_dims, num_cosets
        )
    return _apply_sparse_parallel(
        state,
        indptr,
        indices,
        data,
        offsets,
        comp_strides,
        comp_dims,
        num_cosets,
        n_chunks,
    )


# Walk a super-operator sparsely only when its nonzero fraction is below this,
# where skipping zeros beats the CSR indirection (two-qubit depolarizing sits
# near 0.11, one-qubit near 0.375; a dense gate super-operator is 1.0).
_SPARSE_SUPEROP_FRACTION = 0.5


def _superop_csr(matrix: np.ndarray) -> tuple | None:
    """CSR of ``matrix``'s exact nonzeros, or ``None`` if it is too dense.

    Lists each row's nonzeros in increasing column order, so a sparse walk
    reproduces the dense dot bit for bit - a skipped entry is an exact zero the
    dense sum would have added as ``0.0``, in the same column order. Returns
    ``(indptr, indices, data)``; ``None`` when the nonzero fraction is above
    `_SPARSE_SUPEROP_FRACTION`, where the indirection would not pay off.
    """
    d = matrix.shape[0]
    indptr = np.empty(d + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[complex] = []
    indptr[0] = 0
    for r in range(d):
        row = matrix[r]
        for c in range(d):
            if row[c] != 0.0:
                indices.append(c)
                data.append(complex(row[c]))
        indptr[r + 1] = len(indices)
    if len(indices) > _SPARSE_SUPEROP_FRACTION * d * d:
        return None
    return (
        indptr,
        np.asarray(indices, dtype=np.int64),
        np.asarray(data, dtype=np.complex128),
    )


# Gate-matrix structure codes. A diagonal or (phased) permutation matrix has at
# most one nonzero per row, so its application is a single multiply per amplitude
# instead of the D-term dot product a dense row needs - a machine-independent
# constant-factor win that covers most common gates (Z/S/T/RZ/CPhase/CZ are
# diagonal; X/CX/CCX/SWAP/iSwap are permutations).
_DENSE = 0
_DIAGONAL = 1
_PERMUTATION = 2


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


# Declared structure per canonical gate identity: the key-to-kernel table.
# Which gates share a kernel is decided HERE, per engine - never in the key.
#
# How kernel selection flows (one classifier, one kernel family):
#   standard path   apply(step) -> _resolve_structure(step): a key declared
#                   _DENSE skips the scan entirely; everything else takes ONE
#                   `_classify_matrix` scan at plan preparation, cached per
#                   step - never per application, never per shot.
#   fallback path   _apply_local(state, matrix, targets): matrix-only callers
#                   (reset shifts, noise Kraus branches) - the same single
#                   `_classify_matrix` scan, then the same resolved kernels.
#   fused path      _compile_dynamic_plan bakes each gate's resolved code into
#                   the plan arrays, so the shots kernel never classifies.
#
# An entry may claim _DIAGONAL or _PERMUTATION only if that structure holds
# for EVERY parameter value of the gate (RZ is diagonal for all theta); a
# parametric gate whose structure varies claims _DENSE, which is always safe
# (RX at theta=pi happens to be a permutation, but _DENSE stays correct).
# The spec-vs-content test in test_kernel_dispatch.py enforces agreement
# with `_classify_matrix` over the whole default map.
_K = BuiltinKernelKey
_KERNEL_SPECS: dict[BuiltinKernelKey, int] = {
    # diagonal for every parameter value
    _K.I: _DIAGONAL,
    _K.Z: _DIAGONAL,
    _K.S: _DIAGONAL,
    _K.SDG: _DIAGONAL,
    _K.T: _DIAGONAL,
    _K.TDG: _DIAGONAL,
    _K.CZ: _DIAGONAL,
    _K.CS: _DIAGONAL,
    _K.RZ: _DIAGONAL,
    _K.PHASE: _DIAGONAL,
    _K.CPHASE: _DIAGONAL,
    _K.CLOCK: _DIAGONAL,
    _K.CCLOCK: _DIAGONAL,
    _K.SUBSPACE_RZ: _DIAGONAL,
    # exactly one nonzero per row, for every parameter value
    _K.X: _PERMUTATION,
    _K.Y: _PERMUTATION,
    _K.CX: _PERMUTATION,
    _K.CY: _PERMUTATION,
    _K.SWAP: _PERMUTATION,
    _K.ISWAP: _PERMUTATION,
    _K.CCX: _PERMUTATION,
    _K.CSWAP: _PERMUTATION,
    _K.SHIFT: _PERMUTATION,
    _K.SUM: _PERMUTATION,
    _K.SWAP_LEVELS: _PERMUTATION,
    # generically dense
    _K.H: _DENSE,
    _K.SX: _DENSE,
    _K.RX: _DENSE,
    _K.RY: _DENSE,
    _K.FOURIER: _DENSE,
    _K.FOURIERDG: _DENSE,
    _K.SUBSPACE_RX: _DENSE,
    _K.SUBSPACE_RY: _DENSE,
}


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
def _dense_range(
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
def _diagonal_range(
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
def _permutation_range(
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


@njit(cache=True)
def _dispatch_range(
    state, code, matrix, columns, values, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Route one coset range to the structure-specialized kernel."""
    if code == _DIAGONAL:
        _diagonal_range(state, values, offsets, comp_strides, comp_dims, start, end)
    elif code == _PERMUTATION:
        _permutation_range(
            state, columns, values, offsets, comp_strides, comp_dims, start, end
        )
    else:
        _dense_range(state, matrix, offsets, comp_strides, comp_dims, start, end)


@njit(cache=True)
def _apply_resolved_serial(
    state, code, matrix, columns, values, offsets, comp_strides, comp_dims, num_cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Single-threaded application of an already-classified gate."""
    _dispatch_range(
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
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Parallel application of an already-classified gate over coset chunks."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        start = chunk * num_cosets // n_chunks
        end = (chunk + 1) * num_cosets // n_chunks
        if start < end:
            _dispatch_range(
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
            )
    return state


# --- sparse coset kernel (density-matrix channel super-operators) ---
#
# A channel super-operator ``sum_i kron(K_i, conj(K_i))`` is often mostly zero
# even when neither diagonal nor a permutation (two-qubit depolarizing fills
# ``2 d**2 - d`` of ``d**4`` entries), which `_classify_matrix` calls dense.
# These kernels walk a CSR of the exact nonzeros in column order - bit-identical
# to the dense walk, at a fraction of the multiplies. `NumbaDMEngine` only.


@njit(cache=True)
def _sparse_range(
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


@njit(cache=True)
def _apply_sparse_serial(
    state, indptr, indices, data, offsets, comp_strides, comp_dims, num_cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Single-threaded sparse super-operator application."""
    _sparse_range(
        state, indptr, indices, data, offsets, comp_strides, comp_dims, 0, num_cosets
    )
    return state


@njit(cache=True, parallel=True)
def _apply_sparse_parallel(
    state, indptr, indices, data, offsets, comp_strides, comp_dims, num_cosets, n_chunks
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Parallel sparse super-operator application over coset chunks."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        start = chunk * num_cosets // n_chunks
        end = (chunk + 1) * num_cosets // n_chunks
        if start < end:
            _sparse_range(
                state,
                indptr,
                indices,
                data,
                offsets,
                comp_strides,
                comp_dims,
                start,
                end,
            )
    return state


@njit(cache=True)
def _probabilities_kernel(
    state: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities of a flat statevector.

    Mirrors ``np.abs(state) ** 2`` normalized by its sum: ``abs`` of a complex
    value is ``sqrt(re^2 + im^2)``, squared back to the Born probability.
    """
    size = state.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        magnitude = abs(state[i])
        probabilities[i] = magnitude * magnitude
        total += probabilities[i]
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _normalized_cdf(
    probabilities: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Cumulative distribution normalized by its last element (``cdf /= cdf[-1]``).

    Built with the same sequential cumulative sum NumPy's ``cumsum`` uses, so a
    ``searchsorted`` against it matches ``rng.choice``'s internal inverse-CDF.
    """
    size = probabilities.shape[0]
    cdf = np.empty(size, dtype=np.float64)
    running = 0.0
    for i in range(size):
        running += probabilities[i]
        cdf[i] = running
    last = cdf[size - 1]
    for i in range(size):
        cdf[i] = cdf[i] / last
    return cdf


@njit(cache=True)
def _searchsorted_right(
    cdf: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Index of the first ``cdf`` entry strictly greater than ``u``.

    Equivalent to ``cdf.searchsorted(u, side="right")`` for a non-decreasing
    ``cdf`` - the inverse-CDF step of categorical sampling.
    """
    lo = 0
    hi = cdf.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _sample_index_kernel(
    probabilities: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Draw one flat basis index from ``probabilities`` given uniform ``u``."""
    return _searchsorted_right(_normalized_cdf(probabilities), u)


@njit(cache=True)
def _sample_indices_kernel(
    probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Draw one flat index per uniform in ``uniforms`` (built cdf reused)."""
    cdf = _normalized_cdf(probabilities)
    out = np.empty(uniforms.shape[0], dtype=np.int64)
    for j in range(uniforms.shape[0]):
        out[j] = _searchsorted_right(cdf, uniforms[j])
    return out


@njit(cache=True)
def _project_kernel(
    state: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project onto the measured digits of ``index`` and renormalize.

    Keeps every basis state whose measured subsystem digits match those of the
    sampled ``index``, zeros the rest, then divides by the surviving norm -
    exactly ``np.linalg.norm``-normalized projective collapse.
    """
    size = state.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    out = np.zeros(size, dtype=np.complex128)
    norm_sq = 0.0
    for flat in range(size):
        keep = True
        for j in range(m):
            if (flat // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                keep = False
                break
        if keep:
            amplitude = state[flat]
            out[flat] = amplitude
            norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    if norm_sq > 0.0:
        norm = sqrt(norm_sq)
        for flat in range(size):
            out[flat] = out[flat] / norm
    return out


@njit(cache=True)
def _shift_subsystem(
    state: np.ndarray, stride: int, dim: int, outcome: int
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Cyclically shift one subsystem's digit down by ``outcome`` (reset to |0>).

    Equivalent to applying ``shift_matrix(dim, -outcome)`` to the subsystem: the
    amplitude with subsystem digit ``g`` moves to digit ``(g - outcome) mod dim``.
    After a collapse only digit ``outcome`` is populated, so this relabels the
    measured branch to |0>.
    """
    size = state.shape[0]
    out = np.zeros(size, dtype=np.complex128)
    for flat in range(size):
        digit = (flat // stride) % dim
        shifted = (digit - outcome + dim) % dim
        out[flat + (shifted - digit) * stride] = state[flat]
    return out


@njit(cache=True)
def _condition_passes(
    clbits, cond_clbit, cond_value, start, length
) -> bool:  # pragma: no cover - compiled by Numba
    """Whether the lowered feedforward condition ``clbits[c] == v`` all hold."""
    for t in range(length):
        if clbits[cond_clbit[start + t]] != cond_value[start + t]:
            return False
    return True


@njit(cache=True)
def _apply_step(
    state,
    a,
    ap_mat_ptr,
    ap_dim,
    ap_off_ptr,
    ap_comp_ptr,
    ap_comp_len,
    ap_code,
    mat_flat,
    off_flat,
    col_flat,
    val_flat,
    comp_stride_flat,
    comp_dim_flat,
    size,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply compiled gate ``a`` to ``state`` in place (serial coset walk).

    Structure (``ap_code``/columns/values) was resolved when the plan was
    compiled - by declared kernel key or one content scan - so no per-shot
    classification happens here.
    """
    d = ap_dim[a]
    mat_start = ap_mat_ptr[a]
    matrix = mat_flat[mat_start : mat_start + d * d].reshape(d, d)
    off_start = ap_off_ptr[a]
    offsets = off_flat[off_start : off_start + d]
    comp_start = ap_comp_ptr[a]
    comp_end = comp_start + ap_comp_len[a]
    columns = col_flat[off_start : off_start + d]
    values = val_flat[off_start : off_start + d]
    code = ap_code[a]
    _dispatch_range(
        state,
        code,
        matrix,
        columns,
        values,
        offsets,
        comp_stride_flat[comp_start:comp_end],
        comp_dim_flat[comp_start:comp_end],
        0,
        size // d,
    )


@njit(cache=True)
def _measure_step(
    state,
    clbits,
    m,
    me_ptr,
    me_len,
    me_classical,
    me_stride,
    me_dim,
    me_conf_ptr,
    conf_flat,
    uniforms,
    draw,
):  # pragma: no cover - compiled by Numba
    """Collapse the measured subsystems, write their reported digits to ``clbits``.

    Takes the shot's uniform pool and its draw cursor rather than a single
    uniform, because the number of draws depends on the noise attached to this
    measurement: one for the Born-sampled collapse, then one more for each
    measured subsystem carrying a readout confusion. The collapse always keeps
    the *true* outcome and only the reported classical value is resampled
    (`_report_digit_kernel`), so state export and qubit reuse are untouched
    while feedforward reads the report - the same semantics, and the same draw
    order, as ``_run_one_shot``'s ``_report_digit`` on the NumPy path.

    Returns the projected state and the advanced cursor.
    """
    start = me_ptr[m]
    length = me_len[m]
    index = _sample_index_kernel(_probabilities_kernel(state), uniforms[draw])
    draw += 1
    state = _project_kernel(
        state, me_stride[start : start + length], me_dim[start : start + length], index
    )
    for j in range(length):
        digit = (index // me_stride[start + j]) % me_dim[start + j]
        conf_ptr = me_conf_ptr[start + j]
        if conf_ptr >= 0:
            digit = _report_digit_kernel(
                conf_flat, conf_ptr, me_dim[start + j], digit, uniforms[draw]
            )
            draw += 1
        clbits[me_classical[start + j]] = digit
    return state, draw


@njit(cache=True)
def _reduced_density(
    state, offsets, comp_strides, comp_dims, cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Reduced density matrix ``rho_T`` of the target subsystems, ``d x d``.

    Traces the flat state over the complement (non-target) subsystems:
    ``rho_T[a, b] = sum_cosets psi[base + offsets[a]] * conj(psi[base +
    offsets[b]])``, one pass over every amplitude with the coset kernels'
    division-free odometer. The ``O(size * d)`` object that lets a channel weigh
    all ``num`` Kraus branches without applying any (see `_channel_step`).
    """
    d = offsets.shape[0]
    rho = np.zeros((d, d), dtype=np.complex128)
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(cosets):
        for a in range(d):
            amplitude = state[base + offsets[a]]
            for b in range(d):
                rho[a, b] += amplitude * state[base + offsets[b]].conjugate()
        base = _advance_base(base, comp_strides, comp_dims, counter)
    return rho


@njit(cache=True)
def _channel_step(
    state,
    c,
    ch_kra_ptr,
    ch_num_kraus,
    ch_dim,
    ch_off_ptr,
    ch_comp_ptr,
    ch_comp_len,
    ch_kra_flat,
    ch_mmat_flat,
    ch_off_flat,
    ch_comp_stride_flat,
    ch_comp_dim_flat,
    size,
    u,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Sample one Kraus branch of compiled channel ``c`` (quantum-jump step).

    Weighs the branches the way Aer's trajectory sampler does, without applying
    a non-chosen operator: branch probability ``p_i = <psi|K_i^dagger K_i|psi>``
    equals ``Tr(M_i rho_T)`` over the targets' reduced density matrix ``rho_T``
    (`_reduced_density`; ``M_i = K_i^dagger K_i`` precomputed in the channel
    table), so one reduced density matrix plus a ``d x d`` trace per operator -
    ``O(size * d + num * d**2)`` - replaces materializing and norming ``num``
    full branches. Only the chosen operator is applied, to a fresh copy
    normalized by its own norm, classified in place and routed to its structure
    kernel like a gate. One uniform ``u`` is consumed.

    A mathematically equal but numerically distinct estimator of the same
    channel as the NumPy reference: distributions agree and each engine stays
    seed-reproducible, but per-seed counts are not bit-identical across the two
    (see the module docstring's contract).
    """
    d = ch_dim[c]
    num = ch_num_kraus[c]
    off_start = ch_off_ptr[c]
    offsets = ch_off_flat[off_start : off_start + d]
    comp_start = ch_comp_ptr[c]
    comp_end = comp_start + ch_comp_len[c]
    comp_strides = ch_comp_stride_flat[comp_start:comp_end]
    comp_dims = ch_comp_dim_flat[comp_start:comp_end]
    kra_start = ch_kra_ptr[c]
    cosets = size // d

    rho = _reduced_density(state, offsets, comp_strides, comp_dims, cosets)
    weights = np.empty(num, dtype=np.float64)
    for i in range(num):
        mmat_start = kra_start + i * d * d
        mmat = ch_mmat_flat[mmat_start : mmat_start + d * d].reshape(d, d)
        trace = 0.0
        for a in range(d):
            for b in range(d):
                trace += (mmat[a, b] * rho[b, a]).real
        # Tr(M_i rho_T) is real and PSD-nonnegative; clamp the round-off tail.
        weights[i] = trace if trace > 0.0 else 0.0
    chosen = _inverse_cdf_pick(weights, u)

    out = np.empty(size, dtype=np.complex128)
    for j in range(size):
        out[j] = state[j]
    kraus_start = kra_start + chosen * d * d
    kraus = ch_kra_flat[kraus_start : kraus_start + d * d].reshape(d, d)
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    code = _classify_matrix(kraus, columns, values)
    _dispatch_range(
        out, code, kraus, columns, values, offsets, comp_strides, comp_dims, 0, cosets
    )
    norm_sq = 0.0
    for j in range(size):
        amplitude = out[j]
        norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    norm = sqrt(norm_sq)
    for j in range(size):
        out[j] = out[j] / norm
    return out


@njit(cache=True)
def _reset_step(
    state, r, rs_ptr, rs_len, rs_stride, rs_dim, u
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Collapse the reset subsystems, then shift each measured branch to |0>."""
    start = rs_ptr[r]
    length = rs_len[r]
    index = _sample_index_kernel(_probabilities_kernel(state), u)
    state = _project_kernel(
        state, rs_stride[start : start + length], rs_dim[start : start + length], index
    )
    for j in range(length):
        sub_stride = rs_stride[start + j]
        sub_dim = rs_dim[start + j]
        outcome = (index // sub_stride) % sub_dim
        if outcome != 0:
            state = _shift_subsystem(state, sub_stride, sub_dim, outcome)
    return state


@njit(cache=True, parallel=True)
def _run_shots_kernel(
    # per-step sequencer (one entry per plan step, in program order)
    step_kind,  # 0=gate, 1=measurement, 2=reset
    step_data,  # row index into this kind's table (gate/measure/reset)
    step_cond_ptr,  # start of this step's condition in the condition pool
    step_cond_len,  # number of (clbit, value) terms in the condition
    # condition pool (flat, shared by all conditioned steps)
    cond_clbit,  # clbit each feedforward term reads
    cond_value,  # value that clbit must equal for the step to fire
    # gate table (one entry per ApplyMatrixStep)
    ap_mat_ptr,  # start of the d*d matrix in mat_flat
    ap_dim,  # local dimension d (matrix is d*d, with d offsets)
    ap_off_ptr,  # start of the d local->flat offsets in off_flat
    ap_comp_ptr,  # start of the complement strides/dims in comp_*_flat
    ap_comp_len,  # number of complement (non-target) subsystems
    ap_code,  # structure code per gate, resolved at compile time
    # gate flat backing
    mat_flat,  # concatenated row-major gate matrices (complex128)
    off_flat,  # concatenated local-index -> flat-offset tables
    col_flat,  # per-row nonzero columns (aligned with off_flat; 0-pad if dense)
    val_flat,  # per-row nonzero values (aligned with off_flat; 0-pad if dense)
    comp_stride_flat,  # concatenated complement strides
    comp_dim_flat,  # concatenated complement dimensions
    # measurement table (one entry per MeasurementStep)
    me_ptr,  # start of this measurement's subsystems in me_*
    me_len,  # number of measured subsystems
    # measurement flat backing
    me_classical,  # clbit each measured subsystem's digit is written to
    me_stride,  # flat stride of each measured subsystem
    me_dim,  # dimension of each measured subsystem (also each confusion's side)
    me_conf_ptr,  # start of this subsystem's confusion in conf_flat, -1 if none
    conf_flat,  # concatenated row-major confusion matrices (float64)
    # reset table (one entry per ResetStep)
    rs_ptr,  # start of this reset's subsystems in rs_*
    rs_len,  # number of reset subsystems
    # reset flat backing
    rs_stride,  # flat stride of each reset subsystem
    rs_dim,  # dimension of each reset subsystem
    # channel table (one entry per ApplyChannelStep), built by `noise.nb`
    ch_kra_ptr,  # start of this channel's Kraus stack in ch_kra_flat
    ch_num_kraus,  # number of Kraus operators in the stack
    ch_dim,  # local dimension d (each operator is d*d, with d offsets)
    ch_off_ptr,  # start of the d local->flat offsets in ch_off_flat
    ch_comp_ptr,  # start of the complement strides/dims in ch_comp_*_flat
    ch_comp_len,  # number of complement (non-target) subsystems
    # channel flat backing (its own pools: a channel carries a stack, not one matrix)
    ch_kra_flat,  # concatenated row-major Kraus operators (complex128)
    ch_mmat_flat,  # concatenated K_i^dagger K_i per Kraus (complex128), for branch weights
    ch_off_flat,  # concatenated local-index -> flat-offset tables
    ch_comp_stride_flat,  # concatenated complement strides
    ch_comp_dim_flat,  # concatenated complement dimensions
    size,  # statevector length prod(dims); each shot allocates its own buffer
    n_clbits,  # classical-register width: per-shot clbits and result columns
    shots,  # number of independent trajectories - the `prange` extent
    uniforms,  # pre-drawn uniforms, shots*max_draws in execution order
    max_draws,  # per-shot uniform budget; shot s reads uniforms[s*max_draws:]
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Run ``shots`` independent dynamic trajectories in parallel.

    Each shot (a `prange` iteration) owns a private state and classical register
    and interprets the compiled plan: conditioned gate application, projective
    measurement with readout error, conditioned reset, and conditioned channel
    noise. Uniforms are pre-drawn per shot in execution order (slice
    ``uniforms[s*max_draws:]``), consumed one per measurement, one more per
    confusion-bearing measured subsystem, and one per firing reset or channel -
    matching the serial path's RNG stream, so counts are identical. Shots are
    independent, so the result is deterministic regardless of thread scheduling.
    """
    num_steps = step_kind.shape[0]
    results = np.zeros((shots, n_clbits), dtype=np.int64)
    for shot in prange(shots):  # pylint: disable=not-an-iterable
        state = np.zeros(size, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        clbits = np.zeros(n_clbits, dtype=np.int64)
        draw = shot * max_draws

        for st in range(num_steps):
            kind = step_kind[st]
            passes = _condition_passes(
                clbits, cond_clbit, cond_value, step_cond_ptr[st], step_cond_len[st]
            )
            if kind == 1:  # measurement (unconditional)
                state, draw = _measure_step(
                    state,
                    clbits,
                    step_data[st],
                    me_ptr,
                    me_len,
                    me_classical,
                    me_stride,
                    me_dim,
                    me_conf_ptr,
                    conf_flat,
                    uniforms,
                    draw,
                )
            elif kind == 0 and passes:  # gate
                _apply_step(
                    state,
                    step_data[st],
                    ap_mat_ptr,
                    ap_dim,
                    ap_off_ptr,
                    ap_comp_ptr,
                    ap_comp_len,
                    ap_code,
                    mat_flat,
                    off_flat,
                    col_flat,
                    val_flat,
                    comp_stride_flat,
                    comp_dim_flat,
                    size,
                )
            elif kind == 2 and passes:  # reset
                state = _reset_step(
                    state,
                    step_data[st],
                    rs_ptr,
                    rs_len,
                    rs_stride,
                    rs_dim,
                    uniforms[draw],
                )
                draw += 1
            elif kind == 3 and passes:  # channel noise
                state = _channel_step(
                    state,
                    step_data[st],
                    ch_kra_ptr,
                    ch_num_kraus,
                    ch_dim,
                    ch_off_ptr,
                    ch_comp_ptr,
                    ch_comp_len,
                    ch_kra_flat,
                    ch_mmat_flat,
                    ch_off_flat,
                    ch_comp_stride_flat,
                    ch_comp_dim_flat,
                    size,
                    uniforms[draw],
                )
                draw += 1

        for c in range(n_clbits):
            results[shot, c] = clbits[c]
    return results


def _plan_compilable(plan: list) -> bool:
    """Whether the fused dynamic kernel understands every step in the plan.

    The kernel compiles every step type the matrix family lowers today - matrix,
    channel, measurement (including readout error), and reset. It does not yet
    encode non-identity physical-to-reported measurement maps. An unsupported
    step or map must not reach ``_compile_dynamic_plan``; the caller falls back
    to the inherited NumPy per-shot path instead, which executes every step type
    correctly at NumPy speed.
    """
    # pylint: disable-next=fixme
    # TODO: Compile non-identity ``reported_digit_maps`` into the measurement
    # table and apply them after physical collapse but before readout confusion.
    # Until then, keep routing such plans through the semantics-complete NumPy
    # per-shot executor rather than silently reporting the physical digit.
    return all(
        isinstance(
            step, (ApplyMatrixStep, ApplyChannelStep, MeasurementStep, ResetStep)
        )
        and not (
            isinstance(step, MeasurementStep)
            and step.reported_digit_maps is not None
            and any(
                reported_map != tuple(range(len(reported_map)))
                for reported_map in step.reported_digit_maps
            )
        )
        for step in plan
    )


class NumbaSVEngine(NumpySVEngine):
    """State-vector engine with Numba-jitted numeric kernels."""

    def __init__(self, name: str = "numba-sv", config: EngineConfig | None = None):
        super().__init__(name, config)
        # Per-gate layout (offsets/strides/chunk count) keyed by target tuple.
        # The layout depends only on targets and the fixed system dims, so it is
        # reused across gates and shots; `initialize` clears it when dims change.
        self._apply_plans: dict[tuple[int, ...], tuple] = {}
        # Per-step resolved structure (code/columns/values), keyed by id(step)
        # with the step pinned in the value so a recycled id can never alias.
        # Structure is a property of the step's frozen matrix alone - not of
        # system dims - so `initialize` deliberately does not clear it; the
        # per-shot dynamic loop re-initializes per trajectory and must keep
        # its once-per-plan resolutions.
        self._structure_cache: dict[int, tuple] = {}

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        super().initialize(system_dims, n_clbits)
        self._apply_plans = {}

    def run(self, plan, shots, seed, request, *, config: EngineConfig | None = None):
        """Run inside this config's Numba thread scope (see `_thread_scope`)."""
        with _thread_scope(config or self.config):
            return super().run(plan, shots, seed, request, config=config)

    def _resolve_structure(
        self, step: ApplyMatrixStep
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Resolve a step's kernel structure once: the standard-path front end.

        `_classify_matrix` is the single classification rule. A kernel_key
        declared ``_DENSE`` skips the scan entirely (dense handles any matrix
        and its columns/values are never read); every other step - declared
        diagonal/permutation or un-keyed - takes one scan here, at plan
        preparation, cached per step (id-keyed, identity-pinned) instead of
        once per application. For declared codes the scan reproduces the
        declaration (guaranteed for every parameter value by the
        spec-vs-content test); content is always the executable truth.
        """
        cached = self._structure_cache.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        matrix = np.ascontiguousarray(step.matrix, dtype=np.complex128)
        d = matrix.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        if _KERNEL_SPECS.get(step.kernel_key) == _DENSE:
            code = _DENSE  # declared dense: nothing to extract, no scan
        else:
            code = int(_classify_matrix(matrix, columns, values))
        resolved = (code, columns, values)
        self._structure_cache[id(step)] = (step, resolved)
        return resolved

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply one plan step - the standard path (key-aware, cached)."""
        code, columns, values = self._resolve_structure(step)
        self._state = self._launch_resolved(
            self.state, step.matrix, step.target_indices, code, columns, values
        )

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Sample one Kraus branch in Numba (quantum-jump unravelling).

        Each branch ``K_i |psi>`` is built by the local-apply fallback path
        from its own copy of the state (the kernels update their input buffer
        in place, so the source must not be reused), and
        `_jump_branch_kernel` does the channel-specific part: branch weights,
        the pick, the renormalization. Consumes exactly one rng draw, like the
        NumPy twin and like measurement and reset.
        """
        source = self.state
        branches = np.empty((len(step.kraus_ops), source.shape[0]), dtype=np.complex128)
        for i, kraus in enumerate(step.kraus_ops):
            branches[i] = self._apply_local(source.copy(), kraus, step.target_indices)
        self._state = _jump_branch_kernel(branches, float(rng.random()))

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a bare local matrix - the matrix-only fallback path.

        For callers with no step to carry identity or cache against: the
        inherited reset path's shift matrices and noise-channel Kraus
        branches. One `_classify_matrix` scan (the same single rule the
        standard path uses), then the same resolved kernels.
        """
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        d = matrix.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        code = int(_classify_matrix(matrix, columns, values))
        return self._launch_resolved(state, matrix, targets, code, columns, values)

    def _launch_resolved(
        self,
        state: np.ndarray,
        matrix: np.ndarray,
        targets: Sequence[int],
        code: int,
        columns: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Shared kernel launch for the standard and fallback paths."""
        targets = tuple(targets)
        plan = self._apply_plans.get(targets)
        if plan is None:
            plan = self._build_apply_plan(targets)
            self._apply_plans[targets] = plan
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        state = np.ascontiguousarray(state, dtype=np.complex128)
        return _run_resolved(state, matrix, plan, code, columns, values)

    def _build_apply_plan(self, targets: tuple[int, ...]) -> tuple:
        """Strided-block kernel layout for ``targets`` over the physical dims."""
        return _compute_apply_plan(self._dims, targets)

    def _run_per_shot(
        self,
        plan: list,
        shots: int,
        seed: int | None,
        request,
        config: EngineConfig,
    ) -> RawResult:
        """Dynamic execution: fuse the per-shot trajectory, run shots in parallel.

        Counts-only runs compile the plan once and evaluate every shot inside one
        Numba kernel (thread-parallel over shots unless ``config.numba_parallel``
        turns that off - see `run`), replacing the Python per-shot loop and
        per-gate dispatch. State-export runs, plans the kernel cannot represent
        (see `_plan_compilable`), and the no-work case fall back to the serial
        base path, which keeps ``self._state`` for the export.

        ``config``'s ``max_workers`` / ``parallel_mode`` reach only that fallback
        path: they distribute shots across OS processes, which the fused kernel
        does not use.
        """
        state_requested = getattr(request, self._state_field)
        if state_requested or not request.counts or not _plan_compilable(plan):
            return super()._run_per_shot(plan, shots, seed, request, config)

        from .parallel import _shot_seed_sequences

        plan_arrays, max_draws = self._compile_dynamic_plan(plan)
        uniforms = np.empty(shots * max_draws, dtype=np.float64)
        for s, seed_sequence in enumerate(_shot_seed_sequences(seed, shots)):
            uniforms[s * max_draws : (s + 1) * max_draws] = np.random.default_rng(
                seed_sequence
            ).random(max_draws)

        size = prod(self._dims) if self._dims else 1
        rows = _run_shots_kernel(
            *plan_arrays, size, self._n_clbits, shots, uniforms, max_draws
        )
        outcome_keys, outcome_counts = reduce_to_counts(rows)
        return RawResult(
            outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=None
        )

    def _compile_dynamic_plan(self, plan: list) -> tuple[tuple, int]:
        """Flatten a plan into typed arrays for `_run_shots_kernel`.

        Reuses the cached per-target apply layout; measurement/reset carry their
        subsystem strides and dims for in-kernel collapse and reset-shift, while
        the two noise payloads go to ``noise.nb``: a channel hands its Kraus
        stack plus that same layout to `_compile_channel_table`, and each
        measurement hands its confusion tuple to `_compile_readout_table`, whose
        pointers are indexed by the measured-subsystem position built here.

        Returns the kernel's positional plan arrays and the per-shot uniform-draw
        budget: one per measurement, one more per confusion-bearing measured
        subsystem, and one per reset and channel - the upper bound on RNG draws
        (a conditioned reset or channel that never fires simply draws less).
        """
        dims = self._dims
        strides = [prod(dims[:q]) for q in range(len(dims))]

        step_kind: list[int] = []
        step_data: list[int] = []
        step_cond_ptr: list[int] = []
        step_cond_len: list[int] = []
        cond_clbit: list[int] = []
        cond_value: list[int] = []
        ap_mat_ptr, ap_dim, ap_off_ptr, ap_comp_ptr, ap_comp_len = [], [], [], [], []
        ap_code = []
        mat_flat, off_flat, comp_stride_flat, comp_dim_flat = [], [], [], []
        col_flat: list[int] = []
        val_flat: list[complex] = []
        me_ptr, me_len, me_classical, me_stride, me_dim = [], [], [], [], []
        rs_ptr, rs_len, rs_stride, rs_dim = [], [], [], []
        # Noise payloads, flattened in one pass below by the noise package:
        # (kraus_ops, offsets, comp_strides, comp_dims) per channel occurrence,
        # and (num_subsystems, confusions) per measurement - the latter in
        # measurement order, which is the order me_* is filled in.
        channel_entries: list[tuple] = []
        readout_entries: list[tuple] = []
        num_measurements = 0
        num_resets = 0

        def add_condition(condition):
            step_cond_ptr.append(len(cond_clbit))
            step_cond_len.append(0 if condition is None else len(condition))
            for clbit, value in condition or ():
                cond_clbit.append(clbit)
                cond_value.append(value)

        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                offsets, comp_strides, comp_dims, _, _ = self._build_apply_plan(
                    step.target_indices
                )
                step_kind.append(0)
                step_data.append(len(ap_dim))
                add_condition(step.condition)
                ap_dim.append(offsets.shape[0])
                ap_mat_ptr.append(len(mat_flat))
                matrix = np.ascontiguousarray(step.matrix, dtype=np.complex128)
                mat_flat.extend(matrix.ravel().tolist())
                # Structure resolved once at compile time (declared key or a
                # single content scan) instead of per gate per shot in-kernel.
                # col/val ride the same per-gate pointer as the offsets; a
                # dense gate stores zero padding the kernel never reads.
                code, columns, values = self._resolve_structure(step)
                ap_code.append(code)
                local_dim = offsets.shape[0]
                if code == _DENSE:
                    col_flat.extend([0] * local_dim)
                    val_flat.extend([0j] * local_dim)
                else:
                    col_flat.extend(int(c) for c in columns)
                    val_flat.extend(complex(v) for v in values)
                ap_off_ptr.append(len(off_flat))
                off_flat.extend(int(o) for o in offsets)
                ap_comp_ptr.append(len(comp_stride_flat))
                ap_comp_len.append(comp_strides.shape[0])
                comp_stride_flat.extend(int(s) for s in comp_strides)
                comp_dim_flat.extend(int(d) for d in comp_dims)
            elif isinstance(step, MeasurementStep):
                step_kind.append(1)
                step_data.append(len(me_len))
                add_condition(None)
                me_ptr.append(len(me_stride))
                me_len.append(len(step.measured_indices))
                for q in step.measured_indices:
                    me_stride.append(strides[q])
                    me_dim.append(dims[q])
                me_classical.extend(step.classical_indices)
                readout_entries.append((len(step.measured_indices), step.confusions))
                num_measurements += 1
            elif isinstance(step, ApplyChannelStep):
                offsets, comp_strides, comp_dims, _, _ = self._build_apply_plan(
                    step.target_indices
                )
                step_kind.append(3)
                step_data.append(len(channel_entries))
                add_condition(step.condition)
                channel_entries.append(
                    (step.kraus_ops, offsets, comp_strides, comp_dims)
                )
            else:  # ResetStep
                step_kind.append(2)
                step_data.append(len(rs_len))
                add_condition(step.condition)
                rs_ptr.append(len(rs_stride))
                rs_len.append(len(step.reset_indices))
                for q in step.reset_indices:
                    rs_stride.append(strides[q])
                    rs_dim.append(dims[q])
                num_resets += 1

        def i64(values):
            return np.asarray(values, dtype=np.int64)

        me_conf_ptr, conf_flat = _compile_readout_table(readout_entries)
        plan_arrays = (
            i64(step_kind),
            i64(step_data),
            i64(step_cond_ptr),
            i64(step_cond_len),
            i64(cond_clbit),
            i64(cond_value),
            i64(ap_mat_ptr),
            i64(ap_dim),
            i64(ap_off_ptr),
            i64(ap_comp_ptr),
            i64(ap_comp_len),
            i64(ap_code),
            np.asarray(mat_flat, dtype=np.complex128),
            i64(off_flat),
            i64(col_flat),
            np.asarray(val_flat, dtype=np.complex128),
            i64(comp_stride_flat),
            i64(comp_dim_flat),
            i64(me_ptr),
            i64(me_len),
            i64(me_classical),
            i64(me_stride),
            i64(me_dim),
            me_conf_ptr,
            conf_flat,
            i64(rs_ptr),
            i64(rs_len),
            i64(rs_stride),
            i64(rs_dim),
            *_compile_channel_table(channel_entries),
        )
        num_confusions = int(np.count_nonzero(me_conf_ptr >= 0))
        max_draws = (
            num_measurements + num_confusions + num_resets + len(channel_entries)
        )
        return plan_arrays, max_draws

    def probabilities(self) -> np.ndarray:
        return _probabilities_kernel(np.ascontiguousarray(self.state))

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``shots`` flat indices: draw uniforms in NumPy, invert in Numba."""
        return _sample_indices_kernel(self.probabilities(), rng.random(shots))

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project onto it in Numba, return the flat index."""
        index = int(_sample_index_kernel(self.probabilities(), float(rng.random())))
        strides, dims = self._measured_layout(measured_subsystems)
        self._state = _project_kernel(
            np.ascontiguousarray(self.state), strides, dims, index
        )
        return index

    def _measured_layout(
        self, subsystems: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat strides and dimensions of the measured subsystems for the kernel."""
        return _measured_layout(self._dims, subsystems)


# --- density-matrix kernels ---


@njit(cache=True)
def _dm_probabilities_kernel(
    rho: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities from the diagonal of ``rho``.

    Mirrors ``clip(real(diag(rho)), 0, None)`` normalized by its sum: a valid
    density matrix has a real, non-negative diagonal, so tiny negative round-off
    is clamped before normalizing into a sampling distribution.
    """
    size = rho.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        value = rho[i, i].real
        value = max(value, 0.0)
        probabilities[i] = value
        total += value
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _dm_project_kernel(
    rho: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project ``rho`` onto the measured digits of ``index`` and renormalize.

    Keeps the block of basis states whose measured subsystem digits match those
    of the sampled ``index`` (both row and column must match), zeros the rest,
    then divides by the surviving trace - exactly the trace-normalized
    projective collapse ``rho * keep[:, None] * keep[None, :] / Tr``.
    """
    size = rho.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    keep = np.empty(size, dtype=np.bool_)
    for i in range(size):
        match = True
        for j in range(m):
            if (i // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                match = False
                break
        keep[i] = match

    trace = 0.0
    for i in range(size):
        if keep[i]:
            trace += rho[i, i].real
    # Mirror ``new / trace if trace > 0 else new``: on a zero-trace branch leave
    # the (already all-zero) kept block unscaled instead of dividing by zero.
    scale = 1.0 / trace if trace > 0.0 else 1.0

    out = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if keep[i]:
            for j in range(size):
                if keep[j]:
                    out[i, j] = rho[i, j] * scale
    return out


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


class NumbaDMEngine(NumpyDMEngine):
    """Density-matrix engine with Numba-jitted, key-driven numeric kernels.

    Overrides the numeric kernels of `NumpyDMEngine` plus ``run`` (which
    fuses gate/channel pairs and sets the thread scope); strategy selection,
    ``measure_subsystems``, and the fast / per-shot orchestration are inherited
    unchanged and route through the Numba kernels. ``reset_subsystems`` stays
    the inherited NumPy partial-trace channel (see the module docstring).

    Both a gate and a channel apply as one super-operator pass over
    ``vec(rho)`` (module docstring); `_resolve_superop` is the density-matrix
    analog of the statevector `_resolve_structure` - built and classified once
    per plan step, key-aware for gates, content-scanned for channels.
    """

    def __init__(self, name: str = "numba-dm", config: EngineConfig | None = None):
        super().__init__(name, config)
        # Per-target super-operator layout (offsets/strides/chunk count over the
        # doubled bra+ket system), keyed by target tuple. Depends only on
        # targets and the fixed dims; `initialize` clears it when dims change.
        self._sandwich_plans: dict[tuple[int, ...], tuple] = {}
        # Per-step resolved super-operator (matrix/code/columns/values), keyed
        # by id(step) with the step pinned in the value so a recycled id can
        # never alias. A property of the step's frozen payload alone - not of
        # system dims - so `initialize` deliberately does not clear it; the
        # per-shot dynamic loop re-initializes per trajectory and must keep
        # its once-per-plan resolutions.
        self._superop_cache: dict[int, tuple] = {}

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        super().initialize(system_dims, n_clbits)
        self._sandwich_plans = {}

    def run(self, plan, shots, seed, request, *, config: EngineConfig | None = None):
        """Run inside this config's Numba thread scope (see `_thread_scope`).

        Adjacent unconditional gate/channel pairs on the same targets are fused
        into one super-operator first (`_fuse_gate_channels`), halving the
        memory-bound passes a noisy circuit takes.
        """
        with _thread_scope(config or self.config):
            return super().run(
                _fuse_gate_channels(plan), shots, seed, request, config=config
            )

    def _sandwich_plan(self, targets: tuple[int, ...]) -> tuple:
        """Super-operator apply plan for ``targets`` over the doubled dims.

        The super-target combines each gate target's ket subsystem (doubled
        index ``n + t``, stride ``size * prod(dims[:t])``) and bra subsystem
        (doubled index ``t``, stride ``prod(dims[:t])``), ket group first so
        the local index is ``ket * D + bra`` - matching ``kron(M, conj(M))``.
        """
        plan = self._sandwich_plans.get(targets)
        if plan is None:
            n = len(self._dims)
            doubled_dims = self._dims + self._dims
            super_targets = [n + t for t in targets] + list(targets)
            plan = _compute_apply_plan(
                doubled_dims, super_targets, _MIN_SIZE_TO_THREAD_DM
            )
            self._sandwich_plans[targets] = plan
        return plan

    def _resolve_superop(
        self, step: ApplyMatrixStep | ApplyChannelStep
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, tuple | None]:
        """Build and classify a step's super-operator once per plan.

        For a gate the super-operator is ``kron(M, conj(M))``; the Kronecker
        product preserves structure, so a kernel_key declared ``_DENSE`` skips
        the scan and every other gate takes one `_classify_matrix` scan of the
        super-operator. A channel's ``sum_i kron(K_i, conj(K_i))``
        (`noise.nb._kraus_superop_kernel`) carries no key and is always
        scanned - which is what lets a diagonal channel (phase damping) reach
        the diagonal kernel.

        A dense-classified but mostly-zero super-operator (e.g. depolarizing)
        additionally gets a CSR (`_superop_csr`) so the pass can skip its zeros;
        the returned ``sparse`` is that CSR or ``None``. Cached per step
        (id-keyed, identity-pinned), resolved once per plan, not per trajectory.
        """
        cached = self._superop_cache.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        if isinstance(step, ApplyMatrixStep):
            m = np.ascontiguousarray(step.matrix, dtype=np.complex128)
            superop = np.kron(m, m.conj())
            declared_dense = _KERNEL_SPECS.get(step.kernel_key) == _DENSE
        else:
            superop = _kraus_superop_kernel(_kraus_stack(step.kraus_ops))
            declared_dense = False
        d = superop.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        if declared_dense:
            code = _DENSE  # declared dense: nothing to extract, no scan
        else:
            code = int(_classify_matrix(superop, columns, values))
        sparse = _superop_csr(superop) if code == _DENSE else None
        resolved = (superop, code, columns, values, sparse)
        self._superop_cache[id(step)] = (step, resolved)
        return resolved

    def _apply_superop(self, step: ApplyMatrixStep | ApplyChannelStep) -> None:
        """One resolved super-operator pass over ``vec(rho)``, in place.

        ``self.state`` is contiguous ``complex128`` by construction, so the
        flat view aliases it and no per-step copy of the ``4^n`` matrix is
        made. A dense-but-sparse super-operator (a resolved CSR) walks its
        nonzeros; everything else takes the structure-specialized dense /
        diagonal / permutation pass.
        """
        superop, code, columns, values, sparse = self._resolve_superop(step)
        rho = self.state
        flat = np.ascontiguousarray(rho, dtype=np.complex128).reshape(-1)
        plan = self._sandwich_plan(tuple(step.target_indices))
        if sparse is not None:
            _run_resolved_sparse(flat, sparse[0], sparse[1], sparse[2], plan)
        else:
            _run_resolved(flat, superop, plan, code, columns, values)
        self._state = flat.reshape(rho.shape)

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply one gate - key-aware super-operator pass (cached per step)."""
        self._apply_superop(step)

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply the exact Kraus sum as one super-operator pass.

        Deterministic; no randomness is consumed (``rng`` is accepted for
        interface parity, like reset).
        """
        self._apply_superop(step)

    def probabilities(self) -> np.ndarray:
        return _dm_probabilities_kernel(np.ascontiguousarray(self.state))

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``shots`` flat indices: draw uniforms in NumPy, invert in Numba."""
        return _sample_indices_kernel(self.probabilities(), rng.random(shots))

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project onto it in Numba, return the flat index."""
        index = int(_sample_index_kernel(self.probabilities(), float(rng.random())))
        strides, dims = _measured_layout(self._dims, measured_subsystems)
        self._state = _dm_project_kernel(
            np.ascontiguousarray(self.state), strides, dims, index
        )
        return index
