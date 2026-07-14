"""Prototype: in-place Numba kernel for local-matrix application (Plan A).

Statevector-specific optimization variant. Per the backend/engine parallel
handoff, a Numba rewrite of the hot path lives in its own module and does NOT
touch ``NumpyEngine`` or the frozen ``initialize``/``run``/``RawResult``
seam.

This implements the in-place variant of ``numpy_engine._apply_sv`` (the
statevector apply kernel): instead of ``np.tensordot`` + ``np.transpose``
(which allocate ~2x the state per gate), loop over the non-target "rest"
slices, gather each local vector, multiply by the local matrix, and scatter
the result back - no O(N) intermediate allocation.

Conventions match ``_apply_sv`` exactly:

- little-endian flat state: subsystem ``q`` has place value
  ``stride[q] = prod(dims[:q])``.
- local matrix index: ``targets[0]`` is the most-significant local digit,
  ``targets[-1]`` the least-significant. Matrix row/column ``l`` maps to the
  flat state offset ``offsets[l]``.
"""

from __future__ import annotations

from math import prod

import numpy as np
from numba import njit, prange

from .engine_contract import _EngineConfig
from .numpy_engine import NumpyEngine, _apply_sv
from .statevector_backend import StateVectorBackend


@njit(cache=True)
def _apply_matrix_inplace_kernel(state, matrix, offsets, rest_strides, rest_dims, n_bases):
    """Gather/apply/scatter each local block of ``state`` in place.

    For each of the ``n_bases`` non-target slices, the ``dt`` amplitudes selected
    by ``offsets`` are contracted with ``matrix`` (a dense ``dt x dt`` operator)
    and written back. The mixed-radix rest counter advances ``base``
    incrementally, so no per-slice index arithmetic over all subsystems is
    needed. Scratch is two ``dt``-length vectors and one ``n_rest``-length digit
    counter; nothing scales with the full state length ``N``.
    """
    dt = offsets.shape[0]
    n_rest = rest_dims.shape[0]
    vec_in = np.empty(dt, dtype=np.complex128)
    vec_out = np.empty(dt, dtype=np.complex128)
    digits = np.zeros(n_rest, dtype=np.int64)
    base = 0
    for _ in range(n_bases):
        for l in range(dt):
            vec_in[l] = state[base + offsets[l]]
        for o in range(dt):
            row = matrix[o]
            acc = 0.0 + 0.0j
            for i in range(dt):
                acc += row[i] * vec_in[i]
            vec_out[o] = acc
        for o in range(dt):
            state[base + offsets[o]] = vec_out[o]
        # Advance the mixed-radix rest counter, updating `base` in O(1)
        # amortized: each place adds its stride, and a wrapped place subtracts
        # its full span (stride * dim) before carrying into the next.
        for j in range(n_rest):
            digits[j] += 1
            base += rest_strides[j]
            if digits[j] < rest_dims[j]:
                break
            digits[j] = 0
            base -= rest_strides[j] * rest_dims[j]
    return state


@njit(parallel=True, cache=True)
def _apply_matrix_inplace_kernel_parallel(
    state, matrix, offsets, rest_strides, rest_dims, n_bases
):
    """Parallel variant: each non-target slice is an independent ``prange`` task.

    Slices touch disjoint flat positions (fixed ``offsets`` off distinct bases),
    so writing each result back in place races with no other iteration. Unlike
    the serial kernel, ``base`` cannot be carried incrementally across
    iterations - each ``prange`` task decodes its own ``base`` from the
    iteration index (O(n_rest) per slice) and uses a thread-local gather
    buffer.
    """
    dt = offsets.shape[0]
    n_rest = rest_dims.shape[0]
    for r in prange(n_bases):
        # Decode this slice's base flat index from the mixed-radix rest counter.
        # `r` (the prange index) must stay read-only - the parfor pass rejects
        # aliasing/overwriting it - so divide `r` in place of a running counter.
        base = 0
        place = 1
        for j in range(n_rest):
            digit = (r // place) % rest_dims[j]
            base += digit * rest_strides[j]
            place *= rest_dims[j]
        vec_in = np.empty(dt, dtype=np.complex128)
        for l in range(dt):
            vec_in[l] = state[base + offsets[l]]
        for o in range(dt):
            row = matrix[o]
            acc = 0.0 + 0.0j
            for i in range(dt):
                acc += row[i] * vec_in[i]
            state[base + offsets[o]] = acc
    return state


def _plan_inplace(targets, dims):
    """Precompute flat offsets and the rest-counter arrays for one gate.

    Physics-free layout arithmetic, done once per apply on the Python side
    (cost O(n + dt), not O(N)); the O(N) evolution stays inside the jitted
    kernel.

    Returns:
        ``(offsets, rest_strides, rest_dims, n_bases)`` where ``offsets[l]`` is
        the flat state offset of local index ``l`` from a slice base, and the
        rest arrays drive the non-target slice enumeration.
    """
    n = len(dims)
    k = len(targets)
    strides = [1] * n
    for q in range(1, n):
        strides[q] = strides[q - 1] * dims[q - 1]

    local_dims = [dims[t] for t in targets]
    dt = prod(local_dims)
    # Local place values with targets[0] most-significant, matching the matrix's
    # row-major (out_0, out_1, ...) flattening.
    local_strides = [1] * k
    for j in range(k - 2, -1, -1):
        local_strides[j] = local_strides[j + 1] * local_dims[j + 1]
    offsets = np.empty(dt, dtype=np.int64)
    for l in range(dt):
        off = 0
        for j in range(k):
            digit = (l // local_strides[j]) % local_dims[j]
            off += digit * strides[targets[j]]
        offsets[l] = off

    target_set = set(targets)
    rest = [q for q in range(n) if q not in target_set]
    rest_strides = np.array([strides[q] for q in rest], dtype=np.int64)
    rest_dims = np.array([dims[q] for q in rest], dtype=np.int64)
    n_bases = prod(dims[q] for q in rest)
    return offsets, rest_strides, rest_dims, n_bases


def apply_matrix_inplace(state, matrix, targets, dims):
    """Apply a local ``matrix`` to flat ``targets`` of ``state``, mutating in place.

    Drop-in numerics for ``numpy_engine._apply_sv``, but evolves the
    caller's buffer directly instead of returning a new array. ``state`` must be
    a C-contiguous ``complex128`` array (the engine's owned buffer already is);
    ``matrix`` is coerced to contiguous ``complex128``.

    Returns ``state`` for call-site convenience; the same object is mutated.
    """
    assert state.flags.c_contiguous and state.dtype == np.complex128
    if not targets:
        return state
    offsets, rest_strides, rest_dims, n_bases = _plan_inplace(tuple(targets), tuple(dims))
    mat = np.ascontiguousarray(matrix, dtype=np.complex128)
    _apply_matrix_inplace_kernel(state, mat, offsets, rest_strides, rest_dims, n_bases)
    return state


def apply_matrix_inplace_parallel(state, matrix, targets, dims):
    """Multi-threaded (``prange``) counterpart of ``apply_matrix_inplace``.

    Same in-place semantics and conventions; parallelizes across the
    non-target slices. Worth it only when there are enough slices to amortize
    thread dispatch (large ``N``).
    """
    assert state.flags.c_contiguous and state.dtype == np.complex128
    if not targets:
        return state
    offsets, rest_strides, rest_dims, n_bases = _plan_inplace(tuple(targets), tuple(dims))
    mat = np.ascontiguousarray(matrix, dtype=np.complex128)
    _apply_matrix_inplace_kernel_parallel(
        state, mat, offsets, rest_strides, rest_dims, n_bases
    )
    return state


# State length N at or above which the parallel Numba kernel replaces the
# BLAS-backed tensordot `_apply_sv`. Below it, `tensordot` wins (in-cache,
# SIMD-vectorized); at/above it the in-place, zero-allocation `prange` kernel
# wins by ~1.2x rising to ~20x as N grows (see the microbenchmark in
# `progress/`). N = prod(dims) is fixed for a run, so this selects one kernel
# per execution. 2**18 (state ~4 MB, ~18 qubits) is where the parallel kernel
# first overtakes tensordot on the reference machine; it is a machine/cache
# dependent heuristic, like `parallel._PARALLEL_MIN_SHOTS`.
_NUMBA_PARALLEL_MIN_SIZE = 1 << 18


class NumbaEngine(NumpyEngine):
    """`NumpyEngine` with a size-dispatched Numba statevector apply kernel.

    The `NumpyEngine` binds one ``_apply_kernel`` per state semantics and never
    branches on semantics afterwards (see its module docstring, which
    anticipates exactly this "NumbaEngine sibling"). For ``"statevector"``
    semantics this subclass swaps that bound kernel for a size-dispatched one:
    for a state of length ``>= numba_min_size`` it applies each gate with the
    in-place, multi-threaded ``prange`` kernel (`apply_matrix_inplace_parallel`);
    below that, where BLAS-backed ``tensordot`` is faster, it falls back to the
    stock `_apply_sv`. For ``"density_matrix"`` semantics nothing changes - the
    Numba path only covers statevector - so the class is safe for both.

    Because the swap is at the ``_apply_kernel`` seam, every code path that
    applies matrices through `apply` picks it up: the fast path and the *serial*
    per-shot dynamic path. The *parallel* per-shot path is the one exception -
    `parallel.py` constructs a plain `NumpyEngine` in each worker from the
    semantics string (not from this class), so parallel dynamic counts still run
    on the NumPy kernel. Optimizing that path would mean changing the shared
    ``parallel.py`` worker construction and is deliberately out of scope here.

    Results match `NumpyEngine` to floating-point tolerance, not bit-for-bit:
    the two kernels sum in different orders, so counts for a fixed seed are
    reproducible per engine, not across engines (the same relaxation
    `NumpyEngine` already documents across state semantics).

    Reset's shift-back still routes through `_apply_sv` (inside `_reset_sv`); it
    is a rare single-subsystem operation, left on the reference path.
    """

    def __init__(
        self,
        config: _EngineConfig | None = None,
        *,
        state_semantics: str,
        numba_min_size: int = _NUMBA_PARALLEL_MIN_SIZE,
    ) -> None:
        """Create an uninitialized engine with a matrix-kernel size threshold."""
        super().__init__(config, state_semantics=state_semantics)
        self._numba_min_size = int(numba_min_size)
        if state_semantics == "statevector":
            self._apply_kernel = self._sv_apply_size_dispatched

    def _sv_apply_size_dispatched(self, state, matrix, targets, dims, reversed_dims=None):
        """Statevector apply kernel dispatched on state size (`_apply_kernel`)."""
        if state.size >= self._numba_min_size:
            return apply_matrix_inplace_parallel(state, matrix, targets, dims)
        return _apply_sv(state, matrix, targets, dims, reversed_dims)


class StateVectorBackendNumba(StateVectorBackend):
    """`StateVectorBackend` that runs on a `NumbaEngine`.

    `_MatrixBackendBase` constructs its `NumpyEngine` directly (there is no
    engine-construction seam to override), so this backend lets the base build
    normally and then replaces the engine with a `NumbaEngine` carrying the same
    normalized config. All lowering, validation, result assembly, and the public
    `run` contract are inherited. ``numba_min_size`` forwards the engine's
    matrix-kernel threshold (mainly a test/tuning knob; the default suits real
    use).
    """

    def __init__(
        self,
        options: dict | None = None,
        implementation_map=None,
        *,
        numba_min_size: int = _NUMBA_PARALLEL_MIN_SIZE,
    ) -> None:
        super().__init__(options=options, implementation_map=implementation_map)
        # Swap the base's NumpyEngine for a NumbaEngine using the same config.
        # _engine_system stays None (set by the base __init__), so the next run
        # initializes this engine as usual.
        self._engine = NumbaEngine(
            self._engine._config,
            state_semantics=self._state_field,
            numba_min_size=int(numba_min_size),
        )
