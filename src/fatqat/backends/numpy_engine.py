"""NumPy engine for the matrix backend family: one stateful simulator class.

`NumpyEngine` owns the quantum state for both matrix-family state semantics.
The semantics is fixed at construction (``state_semantics``); the
per-semantics numeric kernels are bound once in ``__init__`` and the public
methods never branch on semantics afterwards.

Module structure (relevant for kernel optimization work, e.g. a future
NumbaEngine sibling):

- ``NumpyEngine``: state lifecycle, execution-strategy selection, and the
  fast / per-shot run paths. Semantics-agnostic orchestration only.
- Shared plumbing: plan analysis and the per-shot dynamic loop.
- Shared numeric primitives: ``_contract_local`` (the local matrix
  contraction both apply kernels are built on), ``_measured_keep_mask``
  (the projector mask both collapse kernels are built on), ``_strides``,
  ``_digit``.
- Statevector kernels (``*_sv``) and density-matrix kernels (``*_dm``) with
  matching signatures. Every kernel is a module-level pure function of
  arrays and plain tuples - no engine state, individually replaceable.

Conventions:

- little-endian: flat basis index digit for subsystem ``q`` has place value
  ``prod(dims[:q])``; subsystem 0 is the least-significant digit.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
- a density matrix reshaped to ``reversed(dims) + reversed(dims)`` exposes
  ket axes ``[0, n)`` and bra axes ``[n, 2n)``; subsystem ``q``'s ket axis is
  ``n - 1 - q`` and its bra axis is ``2n - 1 - q``.

Semantics differences (per
``design/architecture/backend/matrix-family/matrix-engine.md``):

- statevector evolution is ``|psi'> = U_T |psi>``; density-matrix evolution
  is ``rho' = U_T rho U_T^dagger``. Neither materializes the global ``U_T``.
- measurement is trajectory-style on both: Born sampling, projective
  collapse, normalization.
- reset samples a branch on the statevector (consumes one rng draw) but is
  the deterministic partial-trace channel on the density matrix (consumes
  none), so reset alone forces per-shot execution only for statevector
  semantics. Consequently, per-shot rng streams are not draw-for-draw
  aligned across semantics: counts for a fixed seed are reproducible per
  backend, not across backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ..implementation.matrices import shift_matrix
from ..result import decode_indices_to_clbit_rows, reduce_to_counts
from .engine_contract import (
    _DensityMatrixResultRequest,
    _EngineConfig,
    _StateVectorResultRequest,
    RawResult,
)
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep, ResolvedStep

_STATE_SEMANTICS = ("statevector", "density_matrix")


class NumpyEngine:
    """Stateful numerical core for statevector or density-matrix evolution.

    The engine owns the current state buffer. Backends initialize the state,
    apply resolved matrix payloads, then sample, collapse, reset, or export
    copies of the state. The request object passed to ``run`` must carry a
    boolean field named after this engine's ``state_semantics``
    (``_StateVectorResultRequest.statevector`` / ``_DensityMatrixResultRequest.density_matrix``).
    """

    def __init__(
        self,
        config: _EngineConfig | None = None,
        *,
        state_semantics: str,
    ) -> None:
        """Create an uninitialized engine with fixed state semantics.

        Args:
            config: Optional execution-strategy options (worker counts,
                parallel mode) for the dynamic counts path.
            state_semantics: ``"statevector"`` or ``"density_matrix"``.
        """
        if state_semantics not in _STATE_SEMANTICS:
            raise ValueError(
                f"unsupported state_semantics={state_semantics!r}; "
                f"expected one of {_STATE_SEMANTICS}"
            )
        self._config = config if config is not None else _EngineConfig()
        self._state_semantics = state_semantics
        # Per-semantics kernels and facts, bound once. This block is the
        # single dispatch point: public methods below call the bound kernels
        # and never branch on semantics themselves.
        if state_semantics == "statevector":
            self._allocate = _allocate_sv
            self._apply_kernel = _apply_sv
            self._probabilities_kernel = _probabilities_sv
            self._collapse_kernel = _collapse_sv
            self._reset_kernel = _reset_sv
            # A pure state cannot hold the mixed post-reset ensemble, so
            # _reset_sv samples a branch (one rng draw); any reset therefore
            # requires per-shot replay.
            self._reset_forces_dynamic = True
        else:
            self._allocate = _allocate_dm
            self._apply_kernel = _apply_dm
            self._probabilities_kernel = _probabilities_dm
            self._collapse_kernel = _collapse_dm
            self._reset_kernel = _reset_dm
            # rho holds the full ensemble, so _reset_dm is the deterministic
            # partial-trace channel (no rng draw); an unconditional reset can
            # stay on the single-evolution fast path.
            self._reset_forces_dynamic = False
        self._state: np.ndarray | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def n_subsystems(self) -> int:
        """Number of subsystems in the currently initialized state."""
        return len(self._dims)

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions and reset to the all-zero computational state."""
        dims = tuple(int(d) for d in system_dims)
        self._state = self._allocate(prod(dims) if dims else 1)
        self._dims = dims
        # Cached once per circuit execution (not per gate): dims are fixed for
        # the engine's lifetime between initialize() calls, so recomputing this
        # reshape shape on every apply() would be pure per-call Python overhead.
        self._reversed_dims = tuple(reversed(dims))
        self._n_clbits = int(n_clbits)

    def apply(self, step: ApplyMatrixStep) -> None:
        """Evolve the state by one resolved matrix step in place."""
        self._require_state()
        self._state = self._apply_kernel(
            self._state,
            step.matrix,
            step.target_indices,
            self._dims,
            self._reversed_dims,
        )

    def probabilities(self) -> np.ndarray:
        """Return normalized computational-basis probabilities."""
        self._require_state()
        return self._probabilities_kernel(self._state)

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample flat basis-state indices from the current state.

        Args:
            shots: Number of samples to draw.
            rng: NumPy random generator used for sampling.

        Returns:
            One-dimensional array of sampled flat basis-state indices.
        """
        self._require_state()
        return rng.choice(self._state.shape[0], size=shots, p=self.probabilities())

    def collapse(self, measured_subsystems: Sequence[int], rng: np.random.Generator) -> int:
        """Sample one outcome, project the internal state, return the flat index."""
        self._require_state()
        idx, new = self._collapse_kernel(
            self._state, measured_subsystems, self._dims, rng
        )
        self._state = new
        return idx

    def measure_subsystems(
        self,
        indices: Sequence[int],
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        """Sample and collapse a group of subsystems in one computational-basis event."""
        self._require_state()
        if len(indices) < 1:
            raise ValueError("measure_subsystems requires at least one index")
        flat = self.collapse(indices, rng)
        return tuple(_digit(flat, index, self._dims) for index in indices)

    def measure_subsystem(self, index: int, rng: np.random.Generator) -> int:
        """Sample and collapse a single subsystem in the computational basis.

        Projects the internal state onto the sampled outcome for ``index`` and
        returns that subsystem's measured digit. Consumes exactly one rng draw.
        """
        return self.measure_subsystems((index,), rng)[0]

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Reprepare a group of subsystems in ``|0>``.

        Statevector semantics: measurement-style branch sampling followed by a
        shift back to ``|0>`` - consumes exactly one rng draw, so ``rng`` is
        required. Density-matrix semantics: the deterministic partial-trace
        channel - ``rng`` is accepted for signature parity and never consumed.
        """
        self._require_state()
        if len(indices) < 1:
            raise ValueError("reset_subsystems requires at least one index")
        self._state = self._reset_kernel(
            self._state, tuple(indices), self._dims, self._reversed_dims, rng
        )

    def reset_subsystem(
        self, index: int, rng: np.random.Generator | None = None
    ) -> None:
        """Reprepare a single subsystem in ``|0>`` (see ``reset_subsystems``)."""
        self.reset_subsystems((index,), rng)

    def export_state(self) -> np.ndarray:
        """Return a copy of the current state (statevector or density matrix)."""
        self._require_state()
        return self._state.copy()

    def run(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: _StateVectorResultRequest | _DensityMatrixResultRequest,
    ) -> RawResult:
        """Execute a lowered plan using this engine's configured system."""
        self._require_state()
        rng = np.random.default_rng(seed)
        is_dynamic, measurements = _analyze_plan_for_run(
            plan, self._reset_forces_dynamic
        )
        if is_dynamic:
            return self._run_per_shot(plan, shots, seed, request)
        return self._run_fast(plan, measurements, shots, rng, request)

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        measurements: list[tuple[int, int]],
        shots: int,
        rng: np.random.Generator,
        request: _StateVectorResultRequest | _DensityMatrixResultRequest,
    ) -> RawResult:
        """Evolve once, optionally sample counts, optionally export state.

        The ``ResetStep`` branch is only reachable under density-matrix
        semantics, where an unconditional reset is a deterministic channel;
        statevector semantics routes every reset-bearing plan to the dynamic
        path (``_reset_forces_dynamic``).
        """
        state_requested = getattr(request, self._state_semantics)
        self.initialize(self._dims, self._n_clbits)
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                self.apply(step)
            elif isinstance(step, ResetStep):
                self.reset_subsystems(step.reset_indices, rng)

        outcome_keys: np.ndarray | None = None
        outcome_counts: np.ndarray | None = None
        state: np.ndarray | None = None

        collapsed_index: int | None = None
        if state_requested and measurements:
            collapsed_index = self.collapse([q for q, _c in measurements], rng)

        if request.counts:
            if measurements:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = self.sample_indices(shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            rows = decode_indices_to_clbit_rows(
                indices, measurements, self._dims, self._n_clbits
            )
            outcome_keys, outcome_counts = reduce_to_counts(rows)

        if state_requested:
            state = self.export_state()

        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
            state=state,
        )

    def _run_per_shot(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: _StateVectorResultRequest | _DensityMatrixResultRequest,
    ) -> RawResult:
        """Run dynamic execution one trajectory at a time or via worker batches."""
        from .parallel import (
            _planned_workers,
            _run_dynamic_shots_parallel,
            _shot_seed_sequences,
        )

        state_requested = getattr(request, self._state_semantics)
        n_iters = shots if request.counts else (1 if state_requested else 0)
        seed_sequences = _shot_seed_sequences(seed, n_iters)

        max_workers = (
            None if state_requested else _planned_workers(self._config, request, n_iters)
        )
        if max_workers is not None:
            snapshots = _run_dynamic_shots_parallel(
                self._config,
                plan,
                self._dims,
                self._n_clbits,
                seed_sequences,
                max_workers,
                self._state_semantics,
            )
        else:
            snapshots: list[tuple[int, ...]] = []
            for seed_sequence in seed_sequences:
                rng = np.random.default_rng(seed_sequence)
                self.initialize(self._dims, self._n_clbits)
                snapshots.append(
                    _execute_dynamic_plan_one_shot(self, plan, self._n_clbits, rng)
                )

        outcome_keys: np.ndarray | None = None
        outcome_counts: np.ndarray | None = None
        state: np.ndarray | None = None

        if request.counts:
            rows = np.asarray(snapshots, dtype=int).reshape((len(snapshots), self._n_clbits))
            outcome_keys, outcome_counts = reduce_to_counts(rows)
        if state_requested:
            state = self.export_state()

        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
            state=state,
        )

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("engine not initialized; call initialize(dims) first")


# --- shared plumbing: plan analysis and the per-shot dynamic loop ---


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None,
    clbits: list[int],
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    return condition is None or all(clbits[c] == v for c, v in condition)


def _execute_dynamic_plan_one_shot(
    engine: NumpyEngine,
    plan: list[ResolvedStep],
    n_clbits: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Run one dynamic-path shot and return its final clbit snapshot.

    Single copy shared by the serial path and `parallel.py`'s worker batches,
    for either state semantics. It touches the engine only through ``apply``,
    ``measure_subsystems``, and ``reset_subsystems`` (reset consumes rng only
    under statevector semantics).
    """
    clbits = [0] * n_clbits
    for step in plan:
        if isinstance(step, ApplyMatrixStep):
            if _condition_matches(step.condition, clbits):
                engine.apply(step)
        elif isinstance(step, MeasurementStep):
            bits = engine.measure_subsystems(step.measured_indices, rng)
            for c, bit in zip(step.classical_indices, bits):
                clbits[c] = bit
        else:
            if _condition_matches(step.condition, clbits):
                engine.reset_subsystems(step.reset_indices, rng)
    return tuple(clbits)


def _analyze_plan_for_run(
    plan: list[ResolvedStep],
    reset_forces_dynamic: bool,
) -> tuple[bool, list[tuple[int, int]]]:
    """Return the dynamic-path decision and fast-path measurement pairs.

    A ``ResetStep`` forces the dynamic path when ``reset_forces_dynamic``
    (statevector semantics: reset samples a branch), and otherwise only when
    it is conditioned or touches an already-measured subsystem (end-of-run
    sampling for that subsystem would otherwise read the post-reset state).
    """
    measured_subsystems: set[int] = set()
    measurements: list[tuple[int, int]] = []
    is_dynamic = False
    for step in plan:
        if isinstance(step, MeasurementStep):
            measured_subsystems.update(step.measured_indices)
            measurements.extend(zip(step.measured_indices, step.classical_indices))
            continue
        if isinstance(step, ResetStep):
            if (
                reset_forces_dynamic
                or step.condition is not None
                or any(t in measured_subsystems for t in step.reset_indices)
            ):
                is_dynamic = True
            continue
        if isinstance(step, ApplyMatrixStep):
            if step.condition is not None:
                is_dynamic = True
            if any(t in measured_subsystems for t in step.target_indices):
                is_dynamic = True
    return is_dynamic, measurements


def _requires_dynamic_execution(
    plan: list[ResolvedStep],
    reset_forces_dynamic: bool,
) -> bool:
    """Return whether the engine must use per-shot execution for the plan."""
    is_dynamic, _measurements = _analyze_plan_for_run(plan, reset_forces_dynamic)
    return is_dynamic


# --- shared numeric primitives ---


def _strides(dims: Sequence[int]) -> list[int]:
    """Little-endian place values: stride[q] = prod(dims[:q])."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def _digit(flat: int, index: int, dims: Sequence[int]) -> int:
    stride = 1
    for q in range(index):
        stride *= dims[q]
    return (flat // stride) % dims[index]


def _contract_local(
    m: np.ndarray,
    tensor: np.ndarray,
    axes: Sequence[int],
    total: int,
    k: int,
) -> np.ndarray:
    """Contract a local ``[out, in]`` operator into ``axes`` of a tensor.

    ``m`` has ``2k`` axes (k out, then k in); ``tensor`` has ``total`` axes.
    The k input axes of ``m`` are contracted with ``axes`` and the resulting
    out axes are permuted back into their positions. This is the single
    matrix-application core both apply kernels are built on: the statevector
    kernel calls it once on the n-axis state tensor, the density-matrix
    kernel twice (ket then bra) on the 2n-axis rho tensor.

    Complexity: one contraction is O(prod(local_dims) * tensor.size) FLOPs
    with ~2x-tensor peak memory (``tensordot`` allocates a new tensor and
    ``transpose`` may copy it again to reorder axes). An in-place variant
    would drop the intermediate allocation; deferred to kernel optimization.
    """
    out = np.tensordot(m, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    # Result axes: [out_0..out_{k-1}] + remaining tensor axes (original order).
    remaining = [ax for ax in range(total) if ax not in axes]
    perm = [0] * total
    for j, ax in enumerate(axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    return np.transpose(out, perm)


def _measured_keep_mask(
    idx: int,
    subsystems: Sequence[int],
    dims: Sequence[int],
    size: int,
) -> np.ndarray:
    """Boolean mask over flat basis states matching ``idx``'s measured digits.

    The projector core both collapse kernels share: an entry is kept when
    every measured subsystem's little-endian digit equals the corresponding
    digit of the sampled index ``idx``.
    """
    subsystems = list(subsystems)
    if len(set(subsystems)) == len(dims):
        keep = np.zeros(size, dtype=bool)
        keep[idx] = True
        return keep
    strides = _strides(dims)
    basis = np.arange(size)
    # One (N, m) broadcast instead of a Python loop of m separate O(N)
    # passes: digits/idx_digits below fold every measured subsystem's stride
    # and modulus into a single vectorized divide/mod/compare.
    stride_arr = np.array([strides[q] for q in subsystems])
    dim_arr = np.array([dims[q] for q in subsystems])
    digits = (basis[:, None] // stride_arr) % dim_arr
    idx_digits = (idx // stride_arr) % dim_arr
    return np.all(digits == idx_digits, axis=1)


# --- statevector kernels ---


def _allocate_sv(size: int) -> np.ndarray:
    """Return the all-zero computational statevector ``|0...0>``."""
    state = np.zeros(size, dtype=complex)
    state[0] = 1.0
    return state


def _apply_sv(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply a local matrix to flat ``targets`` of a little-endian mixed-radix state.

    The matrix's local index treats ``targets[0]`` as the MSB and
    ``targets[k-1]`` as the LSB, sized by each target's own radix from
    ``dims``.

    ``reversed_dims`` is ``tuple(reversed(dims))``, the state's reshape shape.
    Callers that invoke this once per gate on a fixed ``dims`` (the engine's
    hot path) should precompute and pass it; direct/test callers may omit it
    and it is derived from ``dims`` at O(n) cost.
    """
    n = len(dims)
    k = len(targets)
    local_dims = [dims[t] for t in targets]
    if reversed_dims is None:
        reversed_dims = tuple(dims[n - 1 - p] for p in range(n))
    psi = state.reshape(tuple(reversed_dims))
    target_axes = [n - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape(tuple(local_dims) + tuple(local_dims))
    psi = _contract_local(m, psi, target_axes, n, k)
    return psi.reshape(-1)


def _probabilities_sv(state: np.ndarray) -> np.ndarray:
    """Return normalized computational-basis probabilities for a statevector."""
    probabilities = np.abs(state) ** 2
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _collapse_sv(
    state: np.ndarray,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Sample one computational-basis outcome and return the projected state."""
    idx = int(rng.choice(len(state), p=_probabilities_sv(state)))
    keep = _measured_keep_mask(idx, measured_subsystems, dims, len(state))
    new = state.copy()
    new[~keep] = 0.0
    norm = np.linalg.norm(new)
    if norm > 0:
        new = new / norm
    return idx, new


def _reset_sv(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Measure a group of subsystems and reprepare them in ``|0>``.

    Samples one grouped outcome (exactly one rng draw), projects, and shifts
    each nonzero target back to ``|0>`` (``Shift(-k)`` by that target's own
    radix). The rest of an entangled state is left correctly conditioned on
    the sampled branch.
    """
    idx, state = _collapse_sv(state, indices, dims, rng)
    for index in indices:
        outcome = _digit(idx, index, dims)
        if outcome != 0:
            inv = shift_matrix(dims[index], -outcome)
            state = _apply_sv(state, inv, (index,), dims, reversed_dims)
    return state


# --- density-matrix kernels ---


def _allocate_dm(size: int) -> np.ndarray:
    """Return the all-zero computational density matrix ``|0...0><0...0|``."""
    state = np.zeros((size, size), dtype=complex)
    state[0, 0] = 1.0
    return state


def _apply_dm(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply ``rho' = U_T rho U_T^dagger`` for a local matrix on flat ``targets``.

    The full ``prod(dims)``-dimensional ``U_T`` is never materialized: ``rho``
    is viewed as a ``2n``-axis ket/bra tensor and ``U`` (resp. ``conj(U)``) is
    contracted into the target ket (resp. bra) axes via ``_contract_local``.
    """
    n = len(dims)
    k = len(targets)
    local_dims = [dims[t] for t in targets]
    if reversed_dims is None:
        reversed_dims = tuple(dims[n - 1 - p] for p in range(n))
    tensor = rho.reshape(tuple(reversed_dims) * 2)
    m = np.asarray(matrix, dtype=complex).reshape(tuple(local_dims) + tuple(local_dims))
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    tensor = _contract_local(m, tensor, ket_axes, 2 * n, k)
    tensor = _contract_local(m.conj(), tensor, bra_axes, 2 * n, k)
    return tensor.reshape(rho.shape)


def _probabilities_dm(rho: np.ndarray) -> np.ndarray:
    """Return normalized computational-basis probabilities for a density matrix.

    The diagonal of a valid ``rho`` is real and non-negative; tiny negative
    round-off is clipped so the values remain a valid sampling distribution.
    """
    probabilities = np.clip(np.real(np.diagonal(rho)), 0.0, None)
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _collapse_dm(
    rho: np.ndarray,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Sample one computational-basis outcome and return the projected state.

    Born rule on ``diag(rho)``, then ``rho' = P rho P / p`` where ``P`` keeps
    every flat basis state whose measured digits match the sampled outcome.
    """
    size = rho.shape[0]
    idx = int(rng.choice(size, p=_probabilities_dm(rho)))
    keep = _measured_keep_mask(idx, measured_subsystems, dims, size)
    new = rho * keep[:, None] * keep[None, :]
    trace = np.real(np.trace(new))
    if trace > 0:
        new = new / trace
    return idx, new


def _reset_dm(
    rho: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Deterministically reset flat ``targets``: partial trace, then reprepare.

    Reference update from ``density-matrix-backend.md`` §5:
    ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)``. Working view is the
    ``(dt, dt, rest, rest)`` block structure of ``matrix-engine.md`` §5.2; the
    discard step traces the target ket/bra pair and the reprepare step writes
    the reduced state into the all-zero target block. Trace-preserving, so no
    renormalization is needed, and no randomness is consumed (``rng`` is
    accepted only for kernel-signature parity).
    """
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    dt = prod(local_dims)
    rest = rho.shape[0] // dt

    tensor = rho.reshape(tuple(reversed_dims) * 2)
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    moved = ket_axes + bra_axes
    # `remaining` keeps ascending axis order, so rest-ket axes stay contiguous
    # before rest-bra axes and the (dt, dt, rest, rest) regroup below is valid.
    remaining = [ax for ax in range(2 * n) if ax not in moved]
    block = np.transpose(tensor, moved + remaining).reshape(dt, dt, rest, rest)

    rho_rest = np.trace(block, axis1=0, axis2=1)
    post = np.zeros_like(block)
    post[0, 0] = rho_rest

    rest_shape = tuple((tuple(reversed_dims) * 2)[ax] for ax in remaining)
    post = post.reshape(local_dims + local_dims + rest_shape)
    inverse_perm = np.argsort(moved + remaining)
    return np.transpose(post, inverse_perm).reshape(rho.shape)
