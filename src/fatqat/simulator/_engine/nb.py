"""Numba matrix-family engines.

`NumbaSVEngine` (statevector) and `NumbaDMEngine` (density matrix) reuse
every semantics-agnostic piece of their NumPy twins - strategy selection, the
fast and per-shot paths, ``initialize`` / ``measure_subsystems`` dispatch - and
replace only the numeric kernels with Numba-jitted loops. `NumbaUnitaryEngine`
and `NumbaSuperopEngine` then reuse *those* kernels for the operator
representations, overriding only the apply plan (see the operator-engine
section at the bottom). All four are reachable via
``Simulator(method=..., runtime="numba")``.

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
statevector, the channel super-operator for the density matrix, readout-confusion
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

The operator engines do not apply gates one at a time. An operator buffer is
``(rows, columns)`` and every step acts on the row index alone, so columns are
independent and a whole plan runs inside one parallel region split over column
blocks (`_NumbaOperatorRunMixin.run`). Those kernels keep the per-step
accumulation order, so they are bit-identical to the per-gate coset kernels.
A large plan is gate-fused first (`_fuse_operator_payloads`), merging adjacent
steps into wider ones; that one is equal to the unfused plan only to
floating-point tolerance.

Parallelism has two independent axes: ``EngineConfig``'s ``max_workers`` /
``parallel_mode`` distribute dynamic shots across OS processes (reaching only
the inherited NumPy per-shot path), while ``numba_parallel`` switches this
module's in-process thread parallelism for a whole run (`_thread_scope`, and
for a fused operator run `_operator_chunks`, which skips the region outright).

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
from dataclasses import dataclass
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
from .np import (
    _NumpyOperatorEngine,
    NumpyDMEngine,
    NumpySuperopEngine,
    NumpySVEngine,
    NumpyUnitaryEngine,
)

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

    local_index = np.arange(local_dim, dtype=np.int64)
    offsets = np.zeros(local_dim, dtype=np.int64)
    for j, stride in enumerate(target_strides):
        offsets += ((local_index // local_places[j]) % local_dims[j]) * stride

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
# Set by `_pack_operator_plan` for a step whose super-operator resolved to a
# CSR (`_superop_csr`); `_classify_matrix` never returns it.
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


# --- fused operator kernels (one parallel region for a whole plan) ---
#
# An operator buffer is ``(rows, columns)`` row-major and every step acts on the
# row index alone, so a block of columns is closed under the whole plan and the
# parallel region sits outside the step loop. Offsets and complement strides are
# row-space values pre-multiplied by ``columns``, so ``base + offsets[c] + lo``
# addresses the flat buffer directly.


@njit(cache=True)
def _dense_columns(
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
def _diagonal_columns(
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
def _permutation_columns(
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
def _sparse_columns(
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
            _diagonal_columns(
                state, values, offsets, strides, comp_dims, cosets, scratch, lo, hi
            )
        elif step_code == _PERMUTATION:
            cols = sv_flat_columns[sv_ptr[s] : sv_ptr[s + 1]]
            values = sv_flat_values[sv_ptr[s] : sv_ptr[s + 1]]
            _permutation_columns(
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
            _sparse_columns(
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
            _dense_columns(
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


# Minimum per-thread work (steps x amplitudes) before a fused run goes parallel.
_GRAIN_TO_THREAD_OPERATOR = 1 << 14


def _operator_chunks(
    n_steps: int, size: int, n_columns: int, config: EngineConfig
) -> int:
    """Column-block count for a fused operator run; 1 means run serially."""
    if not config.numba_parallel:
        return 1
    if n_steps * size < _MAX_THREADS * _GRAIN_TO_THREAD_OPERATOR:
        return 1
    return max(1, min(_MAX_THREADS, n_columns))


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
        offsets, strides, dims, cosets, _ = _compute_apply_plan(row_dims, targets)
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
def _reduced_diagonal(
    state, offsets, comp_strides, comp_dims, cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Diagonal of the target subsystems' reduced density matrix, ``d`` reals.

    ``rho_T[a, a] = sum_cosets |psi[base + offsets[a]]|**2`` - the marginal
    probability of each local basis state. One pass over the state, where
    `_reduced_density` does ``d`` accumulations per amplitude to build all
    ``d**2`` entries. Enough whenever every ``M_i`` is diagonal, since then
    ``Tr(M_i rho_T) = sum_a M_i[a, a] rho_T[a, a]``.
    """
    d = offsets.shape[0]
    diagonal = np.zeros(d, dtype=np.float64)
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(cosets):
        for a in range(d):
            amplitude = state[base + offsets[a]]
            diagonal[a] += (
                amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
            )
        base = _advance_base(base, comp_strides, comp_dims, counter)
    return diagonal


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
    ch_cdf_ptr,
    ch_mmat_diag,
    ch_kra_flat,
    ch_mmat_flat,
    ch_off_flat,
    ch_comp_stride_flat,
    ch_comp_dim_flat,
    ch_cdf_flat,
    ch_ident_flat,
    size,
    u,
) -> None:  # pragma: no cover - compiled by Numba
    """Sample one Kraus branch of compiled channel ``c`` and apply it in place.

    Which of two weighings runs is decided once per run by
    `_compile_channel_table`:

    - ``ch_cdf_ptr[c] >= 0`` - **Pauli sampling.** The operators are scaled
      unitaries, so ``p_i = <psi|K_i^dagger K_i|psi>`` does not depend on the
      state and its cdf is precomputed. No amplitude is read to make the draw,
      and a draw landing on the identity - the dominant branch at any physical
      error rate - returns having touched nothing.
    - otherwise - **quantum-jump weighing**: ``p_i`` equals ``Tr(M_i rho_T)``
      over the targets' reduced density matrix ``rho_T`` (`_reduced_density`;
      ``M_i = K_i^dagger K_i`` precomputed in the channel table), which is
      ``O(size * d + num * d**2)`` against materializing and norming ``num``
      full branches. ``ch_mmat_diag[c]`` narrows that to the ``d`` times
      cheaper `_reduced_diagonal` when every ``M_i`` is diagonal.

    Either way the drawn operator is applied in place at unit norm: the
    Pauli-sampled operators arrive pre-scaled, and a jump branch is divided by
    the ``sqrt(p_chosen)`` the weighing already computed - both on the ``d x d``
    operator, so nothing passes over the state to renormalize. The round-off
    that leaves in the norm is absorbed by `_probabilities_kernel` and
    `_project_kernel`.

    Both consume exactly one uniform ``u``, so which weighing a channel takes
    never shifts the shot's RNG stream.

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

    cdf_start = ch_cdf_ptr[c]
    if cdf_start >= 0:
        # Probing the leading entry first is a shortcut, not a special case: it
        # is what a search returns for any u below it, and the dominant branch.
        if u < ch_cdf_flat[cdf_start]:
            chosen = 0
        else:
            lo = 1
            hi = num
            while lo < hi:
                mid = (lo + hi) // 2
                if ch_cdf_flat[cdf_start + mid] <= u:
                    lo = mid + 1
                else:
                    hi = mid
            chosen = lo if lo < num else num - 1
        if ch_ident_flat[cdf_start + chosen] != 0:
            return
        scale = 1.0
    else:
        weights = np.empty(num, dtype=np.float64)
        if ch_mmat_diag[c] != 0:
            diagonal = _reduced_diagonal(
                state, offsets, comp_strides, comp_dims, cosets
            )
            for i in range(num):
                mmat_start = kra_start + i * d * d
                trace = 0.0
                for a in range(d):
                    trace += ch_mmat_flat[mmat_start + a * d + a].real * diagonal[a]
                # Tr(M_i rho_T) is real and PSD-nonnegative; clamp round-off.
                weights[i] = trace if trace > 0.0 else 0.0
        else:
            rho = _reduced_density(state, offsets, comp_strides, comp_dims, cosets)
            for i in range(num):
                mmat_start = kra_start + i * d * d
                mmat = ch_mmat_flat[mmat_start : mmat_start + d * d].reshape(d, d)
                trace = 0.0
                for a in range(d):
                    for b in range(d):
                        trace += (mmat[a, b] * rho[b, a]).real
                weights[i] = trace if trace > 0.0 else 0.0
        chosen = _inverse_cdf_pick(weights, u)
        # The drawn branch's squared norm is its own weight. A zero-weight
        # branch has zero cdf width and can never be drawn.
        scale = 1.0 / sqrt(weights[chosen])

    kraus_start = chosen * d * d + kra_start
    operator = np.empty((d, d), dtype=np.complex128)
    for a in range(d):
        for b in range(d):
            operator[a, b] = ch_kra_flat[kraus_start + a * d + b] * scale
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    code = _classify_matrix(operator, columns, values)
    _dispatch_range(
        state,
        code,
        operator,
        columns,
        values,
        offsets,
        comp_strides,
        comp_dims,
        0,
        cosets,
    )


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
    ch_cdf_ptr,  # start of the fixed branch cdf, -1 if state-dependent
    ch_mmat_diag,  # 1 where every K^dagger K is diagonal: weigh from the marginal
    # channel flat backing (its own pools: a channel carries a stack, not one matrix)
    ch_kra_flat,  # concatenated row-major Kraus operators (complex128)
    ch_mmat_flat,  # concatenated K_i^dagger K_i per Kraus (complex128), for branch weights
    ch_off_flat,  # concatenated local-index -> flat-offset tables
    ch_comp_stride_flat,  # concatenated complement strides
    ch_comp_dim_flat,  # concatenated complement dimensions
    ch_cdf_flat,  # concatenated scaled-unitary branch cdfs, normalized (float64)
    ch_ident_flat,  # 1 where that branch is the identity, so the draw is a no-op
    size,  # statevector length prod(dims); each shot allocates its own buffer
    n_clbits,  # classical-register width: per-shot clbits and result columns
    shots,  # number of independent trajectories - the `prange` extent
    uniforms,  # pre-drawn uniforms, shots*max_draws in execution order
    max_draws,  # per-shot uniform budget; shot s reads uniforms[s*max_draws:]
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Run ``shots`` independent dynamic trajectories in parallel.

    Each shot (a `prange` iteration) owns a private state and classical register
    and interprets the compiled plan: conditioned gate application, projective
    measurement with readout confusion, conditioned reset, and conditioned channel
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
            elif kind == 3 and passes:  # channel noise (applied in place)
                _channel_step(
                    state,
                    step_data[st],
                    ch_kra_ptr,
                    ch_num_kraus,
                    ch_dim,
                    ch_off_ptr,
                    ch_comp_ptr,
                    ch_comp_len,
                    ch_cdf_ptr,
                    ch_mmat_diag,
                    ch_kra_flat,
                    ch_mmat_flat,
                    ch_off_flat,
                    ch_comp_stride_flat,
                    ch_comp_dim_flat,
                    ch_cdf_flat,
                    ch_ident_flat,
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
    channel, measurement (including readout confusion), and reset. It does not yet
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

    def _apply_kraus_jump(
        self, step: ApplyChannelStep, rng: np.random.Generator
    ) -> None:
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


# --- gate fusion ---
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


# --- operator engines ---
#
# An operator is its state twin evolved on many columns at once. `run` takes the
# fused whole-plan path; a single application falls back to the per-gate coset
# kernels over a column-batched apply plan, which is a `columns`-sized
# never-targeted subsystem prepended to the dims.


class _NumbaOperatorRunMixin(_NumpyOperatorEngine):
    """Fused whole-plan execution for the Numba operator engines.

    Leaves supply `_operator_row_dims` and `_operator_payloads`.
    """

    def run(
        self,
        plan: list,
        shots: int,
        seed: int | None,
        request,
        *,
        config: EngineConfig | None = None,
    ) -> RawResult:
        """Evolve the identity operator through ``plan`` in one fused call.

        ``shots`` and ``seed`` are unused, as on the NumPy twin.
        """
        config = config or self.config
        with _thread_scope(config):
            self.initialize(self._dims, self._n_clbits)
            payloads = self._operator_payloads(plan)
            if payloads:
                row_dims = self._operator_row_dims()
                operator = np.ascontiguousarray(self.state, dtype=np.complex128)
                flat = operator.reshape(-1)
                n_columns = operator.shape[1]
                if flat.shape[0] >= _MIN_SIZE_TO_FUSE:
                    payloads = _fuse_operator_payloads(payloads, row_dims)
                packed, scratch_rows = _pack_operator_plan(
                    payloads, row_dims, n_columns
                )
                n_chunks = _operator_chunks(
                    len(payloads), flat.shape[0], n_columns, config
                )
                if n_chunks > 1:
                    _run_operator_plan_parallel(
                        flat, n_columns, packed, scratch_rows, n_chunks
                    )
                else:
                    _run_operator_plan_serial(flat, n_columns, packed, scratch_rows)
                self._state = flat.reshape(operator.shape)
            state = self.export_state() if getattr(request, self._state_field) else None
            return RawResult(state=state)

    def _operator_row_dims(self) -> tuple[int, ...]:
        """Subsystem dimensions the operator's row index decomposes into."""
        raise NotImplementedError

    def _operator_payloads(self, plan: list) -> list[tuple]:
        """Resolve ``plan`` into `_pack_operator_plan` payloads."""
        raise NotImplementedError


# Deep base list by design: the leaf adds only its plan builder and payloads.
class NumbaUnitaryEngine(  # pylint: disable=too-many-ancestors
    _NumbaOperatorRunMixin, NumbaSVEngine, NumpyUnitaryEngine
):
    """Unitary engine with Numba-jitted numeric kernels.

    ``U`` is ``size`` statevector columns over a row index that decomposes into
    the plain system dims; every step is a gate.
    """

    def __init__(self, name: str = "numba-unitary", config: EngineConfig | None = None):
        super().__init__(name, config)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        payloads = []
        for step in plan:
            assert isinstance(
                step, ApplyMatrixStep
            ), "unitary execution accepts only matrix steps"
            code, columns, values = self._resolve_structure(step)
            payloads.append(
                (step.matrix, code, columns, values, None, tuple(step.target_indices))
            )
        return payloads

    def _build_apply_plan(self, targets: tuple[int, ...]) -> tuple:
        """Kernel layout for ``targets`` over the column-batched dims."""
        size = prod(self._dims) if self._dims else 1
        extended = (size,) + self._dims
        return _compute_apply_plan(extended, tuple(1 + t for t in targets))

    def _launch_resolved(
        self,
        state: np.ndarray,
        matrix: np.ndarray,
        targets: Sequence[int],
        code: int,
        columns: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Run the inherited launch over the operator's flat ``(row, column)`` buffer."""
        operator = np.ascontiguousarray(state, dtype=np.complex128)
        flat = super()._launch_resolved(
            operator.reshape(-1), matrix, targets, code, columns, values
        )
        return flat.reshape(operator.shape)


# Deep base list by design: the leaf adds only its plan builder and payloads.
class NumbaSuperopEngine(  # pylint: disable=too-many-ancestors
    _NumbaOperatorRunMixin, NumbaDMEngine, NumpySuperopEngine
):
    """Super-operator engine with Numba-jitted numeric kernels.

    ``S`` is ``size**2`` density-matrix columns over a row index that decomposes
    into the doubled ``bra + ket`` dims. Every step - gate, channel, or reset -
    resolves to one super-operator on the combined ket+bra super-target.
    """

    def __init__(self, name: str = "numba-superop", config: EngineConfig | None = None):
        super().__init__(name, config)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims + self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        n = len(self._dims)
        payloads = []
        for step in _fuse_gate_channels(plan):
            # A reset is the Kraus channel sum_k |0><k|.
            resolved_steps = (
                [self._reset_channel(index) for index in step.reset_indices]
                if isinstance(step, ResetStep)
                else [step]
            )
            for resolved in resolved_steps:
                superop, code, columns, values, sparse = self._resolve_superop(resolved)
                targets = tuple(resolved.target_indices)
                # Ket group first, so the local index is ``ket * D + bra``.
                row_targets = tuple(n + t for t in targets) + targets
                payloads.append((superop, code, columns, values, sparse, row_targets))
        return payloads

    def _sandwich_plan(self, targets: tuple[int, ...]) -> tuple:
        """Super-operator layout for ``targets`` over the column-batched doubled dims."""
        plan = self._sandwich_plans.get(targets)
        if plan is None:
            n = len(self._dims)
            size = prod(self._dims) if self._dims else 1
            extended = (size * size,) + self._dims + self._dims
            super_targets = [1 + n + t for t in targets] + [1 + t for t in targets]
            plan = _compute_apply_plan(extended, super_targets, _MIN_SIZE_TO_THREAD_DM)
            self._sandwich_plans[targets] = plan
        return plan
