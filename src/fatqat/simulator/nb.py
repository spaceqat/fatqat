"""Numba statevector simulator.

`NumbaSVSimulator` is a statevector `Simulator` that reuses every
semantics-agnostic piece of `NumpySVSimulator` - strategy selection, the fast
and per-shot paths, ``initialize`` / ``measure_subsystems`` /
``reset_subsystems`` dispatch - and replaces its numeric kernels with
Numba-jitted loops: gate application (`_apply_local`), probability computation
(`probabilities`), categorical sampling (`sample_indices`), and projective
collapse (`collapse`). Because `measure_subsystems` and `reset_subsystems`
delegate to ``collapse`` / ``_apply_local``, they are fully Numba-routed
without being overridden.

The RNG draw itself stays in NumPy: a ``np.random.Generator`` cannot cross into
Numba nopython code, so uniforms are drawn with the passed ``rng`` and the
inverse-CDF search runs in Numba. Drawing ``rng.random(k)`` and inverse-CDF
sampling consumes the stream identically to ``rng.choice(n, p=...)``, so counts
stay reproducible (per simulator; tiny float-summation differences from NumPy
mean counts are reproducible per simulator, not bit-identical across
simulators - the documented contract, see `np.py`).

This is a correctness-first baseline: kernels are straightforward
``O(size)`` / ``O(size * D)`` loops, not tuned. Numba compiles them lazily on
first call. Numba is an optional dependency (the ``numba`` group), so this
module is never imported from ``fatqat.simulator``'s package ``__init__``;
import it explicitly (``from fatqat.simulator.nb import NumbaSVSimulator``).

Conventions match `np.py`: little-endian flat indexing (subsystem ``q`` has
place value ``prod(dims[:q])``, subsystem 0 least-significant) and a local gate
matrix whose most-significant index digit is ``targets[0]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod, sqrt

import numpy as np
from numba import get_num_threads, njit, prange

from ..backends.engine_contract import _EngineConfig as EngineConfig, RawResult
from ..backends.steps import ApplyMatrixStep, MeasurementStep
from ..result import reduce_to_counts
from .np import NumpySVSimulator

_MAX_THREADS = get_num_threads()
# Below this many amplitudes a parallel region costs more than the gate work it
# saves (the state is small and memory-bandwidth bound), so stay single-threaded.
# A coarse machine-independent guard, not a tuned constant.
_MIN_SIZE_TO_THREAD = 1 << 15


def _plan_chunks(num_cosets: int, size: int) -> int:
    """Parallel chunk count for a gate on a ``size``-amplitude state.

    Returns 1 (the serial kernel, no parallel-region overhead) for states below
    `_MIN_SIZE_TO_THREAD`; otherwise splits the cosets across all worker threads.
    """
    if size < _MIN_SIZE_TO_THREAD:
        return 1
    return max(1, min(_MAX_THREADS, num_cosets))


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
def _apply_local_serial(
    state, matrix, offsets, comp_strides, comp_dims, num_cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Single-threaded gate application (no parallel-region overhead)."""
    columns = np.empty(offsets.shape[0], dtype=np.int64)
    values = np.empty(offsets.shape[0], dtype=np.complex128)
    code = _classify_matrix(matrix, columns, values)
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
def _apply_local_parallel(
    state, matrix, offsets, comp_strides, comp_dims, num_cosets, n_chunks
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Gate application split into ``n_chunks`` coset ranges run in parallel.

    The matrix is classified once (serially); each thread then applies the chosen
    kernel to its own disjoint coset range.
    """
    columns = np.empty(offsets.shape[0], dtype=np.int64)
    values = np.empty(offsets.shape[0], dtype=np.complex128)
    code = _classify_matrix(matrix, columns, values)
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
    mat_flat,
    off_flat,
    comp_stride_flat,
    comp_dim_flat,
    size,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply compiled gate ``a`` to ``state`` in place (serial coset walk)."""
    d = ap_dim[a]
    mat_start = ap_mat_ptr[a]
    matrix = mat_flat[mat_start : mat_start + d * d].reshape(d, d)
    off_start = ap_off_ptr[a]
    offsets = off_flat[off_start : off_start + d]
    comp_start = ap_comp_ptr[a]
    comp_end = comp_start + ap_comp_len[a]
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    code = _classify_matrix(matrix, columns, values)
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
    state, clbits, m, me_ptr, me_len, me_classical, me_stride, me_dim, u
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Collapse the measured subsystems, write their digits into ``clbits``."""
    start = me_ptr[m]
    length = me_len[m]
    index = _sample_index_kernel(_probabilities_kernel(state), u)
    state = _project_kernel(
        state, me_stride[start : start + length], me_dim[start : start + length], index
    )
    for j in range(length):
        digit = (index // me_stride[start + j]) % me_dim[start + j]
        clbits[me_classical[start + j]] = digit
    return state


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
    # gate flat backing
    mat_flat,  # concatenated row-major gate matrices (complex128)
    off_flat,  # concatenated local-index -> flat-offset tables
    comp_stride_flat,  # concatenated complement strides
    comp_dim_flat,  # concatenated complement dimensions
    # measurement table (one entry per MeasurementStep)
    me_ptr,  # start of this measurement's subsystems in me_*
    me_len,  # number of measured subsystems
    # measurement flat backing
    me_classical,  # clbit each measured subsystem's digit is written to
    me_stride,  # flat stride of each measured subsystem
    me_dim,  # dimension of each measured subsystem
    # reset table (one entry per ResetStep)
    rs_ptr,  # start of this reset's subsystems in rs_*
    rs_len,  # number of reset subsystems
    # reset flat backing
    rs_stride,  # flat stride of each reset subsystem
    rs_dim,  # dimension of each reset subsystem
    size,  # statevector length prod(dims); each shot allocates its own buffer
    n_clbits,  # classical-register width: per-shot clbits and result columns
    shots,  # number of independent trajectories - the `prange` extent
    uniforms,  # pre-drawn uniforms, shots*max_draws in execution order
    max_draws,  # per-shot uniform budget; shot s reads uniforms[s*max_draws:]
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Run ``shots`` independent dynamic trajectories in parallel.

    Each shot (a `prange` iteration) owns a private state and classical register
    and interprets the compiled plan: conditioned gate application, projective
    measurement, and conditioned reset. Uniforms are pre-drawn per shot in
    execution order (slice ``uniforms[s*max_draws:]``), consumed one per
    measurement and per firing reset - matching the serial path's RNG stream, so
    counts are identical. Shots are independent, so the result is deterministic
    regardless of thread scheduling.
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
                state = _measure_step(
                    state,
                    clbits,
                    step_data[st],
                    me_ptr,
                    me_len,
                    me_classical,
                    me_stride,
                    me_dim,
                    uniforms[draw],
                )
                draw += 1
            elif kind == 0 and passes:  # gate
                _apply_step(
                    state,
                    step_data[st],
                    ap_mat_ptr,
                    ap_dim,
                    ap_off_ptr,
                    ap_comp_ptr,
                    ap_comp_len,
                    mat_flat,
                    off_flat,
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

        for c in range(n_clbits):
            results[shot, c] = clbits[c]
    return results


class NumbaSVSimulator(NumpySVSimulator):
    """State-vector simulator with Numba-jitted numeric kernels."""

    def __init__(self, name: str = "numba-sv", config: EngineConfig | None = None):
        super().__init__(name, config)
        # Per-gate layout (offsets/strides/chunk count) keyed by target tuple.
        # The layout depends only on targets and the fixed system dims, so it is
        # reused across gates and shots; `initialize` clears it when dims change.
        self._apply_plans: dict[tuple[int, ...], tuple] = {}

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        super().initialize(system_dims, n_clbits)
        self._apply_plans = {}

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a local matrix to flat ``targets`` via the Numba kernel."""
        targets = tuple(targets)
        plan = self._apply_plans.get(targets)
        if plan is None:
            plan = self._build_apply_plan(targets)
            self._apply_plans[targets] = plan
        offsets, comp_strides, comp_dims, num_cosets, n_chunks = plan

        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        state = np.ascontiguousarray(state, dtype=np.complex128)
        if n_chunks <= 1:
            return _apply_local_serial(
                state, matrix, offsets, comp_strides, comp_dims, num_cosets
            )
        return _apply_local_parallel(
            state, matrix, offsets, comp_strides, comp_dims, num_cosets, n_chunks
        )

    def _build_apply_plan(self, targets: tuple[int, ...]) -> tuple:
        """Precompute the strided-block kernel layout for a target tuple.

        Returns ``offsets[c]`` (the flat offset of local index ``c``, with
        ``targets[0]`` most-significant), the flat strides/dimensions of the
        complement (non-target) subsystems the kernel odometers over, the coset
        count, and the parallel chunk count for this many cosets.
        """
        dims = self._dims
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
        n_chunks = _plan_chunks(num_cosets, size)
        return offsets, comp_strides, comp_dims, num_cosets, n_chunks

    def _run_per_shot(
        self,
        plan: list,
        shots: int,
        seed: int | None,
        request,
    ) -> RawResult:
        """Dynamic execution: fuse the per-shot trajectory, run shots in parallel.

        Counts-only runs compile the plan once and evaluate every shot inside one
        Numba kernel (thread-parallel over shots), replacing the Python per-shot
        loop and per-gate dispatch. State-export runs (and the no-work case) fall
        back to the serial base path, which keeps ``self._state`` for the export.
        """
        state_requested = getattr(request, self._state_field)
        if state_requested or not request.counts:
            return super()._run_per_shot(plan, shots, seed, request)

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
        subsystem strides and dims for in-kernel collapse and reset-shift. Returns
        the kernel's positional plan arrays and the per-shot uniform-draw budget
        (one per measurement and per reset - the upper bound on RNG draws).
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
        mat_flat, off_flat, comp_stride_flat, comp_dim_flat = [], [], [], []
        me_ptr, me_len, me_classical, me_stride, me_dim = [], [], [], [], []
        rs_ptr, rs_len, rs_stride, rs_dim = [], [], [], []
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
                num_measurements += 1
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
            np.asarray(mat_flat, dtype=np.complex128),
            i64(off_flat),
            i64(comp_stride_flat),
            i64(comp_dim_flat),
            i64(me_ptr),
            i64(me_len),
            i64(me_classical),
            i64(me_stride),
            i64(me_dim),
            i64(rs_ptr),
            i64(rs_len),
            i64(rs_stride),
            i64(rs_dim),
        )
        return plan_arrays, num_measurements + num_resets

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
        strides = np.array([prod(self._dims[:q]) for q in subsystems], dtype=np.int64)
        dims = np.array([self._dims[q] for q in subsystems], dtype=np.int64)
        return strides, dims
