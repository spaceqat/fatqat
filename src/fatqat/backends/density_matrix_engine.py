"""Density-matrix engine: a stateful simulator that owns the quantum state.

Sibling of ``statevector_engine.py`` (see the backend/engine parallel handoff):
same ``initialize(...)`` / ``run(...)`` seam, same ``RawResult`` shape, but the
owned state is a dense density matrix ``rho`` of shape ``(N, N)`` with
``N = prod(system_dims)``.

Conventions (matching the statevector engine):
- little-endian: flat basis index digit for subsystem ``q`` has place value
  ``prod(dims[:q])``; subsystem 0 is the least-significant digit.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
- ``rho`` reshaped to ``reversed(dims) + reversed(dims)`` exposes ket axes
  ``[0, n)`` and bra axes ``[n, 2n)``; subsystem ``q``'s ket axis is
  ``n - 1 - q`` and its bra axis is ``2n - 1 - q``.

Semantics that differ from the statevector engine (per
``design/architecture/backend/matrix-family/density-matrix-backend.md``):
- ordinary evolution is ``rho' = U_T rho U_T^dagger``.
- measurement is trajectory-style: Born sampling from ``diag(rho)``,
  projector posterior, trace normalization.
- reset is a deterministic channel: partial trace over the targets, then
  repreparation in ``|0><0|``. It consumes no randomness, so reset alone
  neither makes a program stochastic nor forces per-shot execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ..result import decode_indices_to_clbit_rows, reduce_to_counts
from .engine_contract import _DensityMatrixResultRequest, _EngineConfig, RawResult
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep, ResolvedStep


class DensityMatrixEngine:
    """Stateful numerical core for density-matrix evolution.

    The engine owns the current density matrix. Backends initialize the state,
    apply resolved matrix payloads, then sample, collapse, reset, or export
    copies of the state.
    """

    def __init__(self, config: _EngineConfig | None = None) -> None:
        """Create an uninitialized engine."""
        self._config = config if config is not None else _EngineConfig()
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
        size = prod(dims) if dims else 1
        state = np.zeros((size, size), dtype=complex)
        state[0, 0] = 1.0
        self._state = state
        self._dims = dims
        # Cached once per circuit execution (not per gate), same rationale as
        # the statevector engine: dims are fixed between initialize() calls.
        self._reversed_dims = tuple(reversed(dims))
        self._n_clbits = int(n_clbits)

    def apply(self, step: ApplyMatrixStep) -> None:
        """Evolve the state by one resolved matrix step: ``U rho U^dagger``."""
        self._require_state()
        self._state = _apply_matrix_rho(
            self._state,
            step.matrix,
            step.target_indices,
            self._dims,
            self._reversed_dims,
        )

    def probabilities(self) -> np.ndarray:
        """Return normalized computational-basis probabilities (``diag(rho)``)."""
        self._require_state()
        return _probabilities_from_rho(self._state)

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
        idx, new = _collapse_density_matrix(
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
        """Discard a group of subsystems and reprepare them in ``|0><0|``.

        On a density matrix, reset is the deterministic channel
        ``rho' = |0><0|_targets (x) Tr_targets(rho)``: no outcome is sampled and
        ``rng`` is accepted only for signature parity with the statevector
        engine (it is never consumed).
        """
        self._require_state()
        if len(indices) < 1:
            raise ValueError("reset_subsystems requires at least one index")
        self._state = _reset_density_matrix(
            self._state, indices, self._dims, self._reversed_dims
        )

    def reset_subsystem(
        self, index: int, rng: np.random.Generator | None = None
    ) -> None:
        """Discard a single subsystem and reprepare it in ``|0><0|``."""
        self.reset_subsystems((index,), rng)

    def export_state(self) -> np.ndarray:
        """Return a copy of the current density matrix."""
        self._require_state()
        return self._state.copy()

    def run(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: _DensityMatrixResultRequest,
    ) -> RawResult:
        """Execute a lowered plan using this engine's configured system."""
        self._require_state()
        rng = np.random.default_rng(seed)
        is_dynamic, measurements = _analyze_plan_for_run(plan)
        if is_dynamic:
            return self._run_per_shot(plan, shots, seed, request)
        return self._run_fast(plan, measurements, shots, rng, request)

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        measurements: list[tuple[int, int]],
        shots: int,
        rng: np.random.Generator,
        request: _DensityMatrixResultRequest,
    ) -> RawResult:
        """Evolve once, optionally sample counts, optionally export state.

        Unlike the statevector fast path, unconditional resets stay on this
        path: they are applied inline as the deterministic partial-trace
        channel, so the evolved density matrix already holds the exact
        ensemble average.
        """
        self.initialize(self._dims, self._n_clbits)
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                self.apply(step)
            elif isinstance(step, ResetStep):
                self.reset_subsystems(step.reset_indices)

        outcome_keys: np.ndarray | None = None
        outcome_counts: np.ndarray | None = None
        state: np.ndarray | None = None

        collapsed_index: int | None = None
        if request.density_matrix and measurements:
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

        if request.density_matrix:
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
        request: _DensityMatrixResultRequest,
    ) -> RawResult:
        """Run dynamic execution one trajectory at a time or via worker batches."""
        from .parallel import (
            _planned_workers,
            _run_dynamic_shots_parallel,
            _shot_seed_sequences,
        )

        n_iters = shots if request.counts else (1 if request.density_matrix else 0)
        seed_sequences = _shot_seed_sequences(seed, n_iters)

        max_workers = (
            None
            if request.density_matrix
            else _planned_workers(self._config, request, n_iters)
        )
        if max_workers is not None:
            snapshots = _run_dynamic_shots_parallel(
                self._config,
                plan,
                self._dims,
                self._n_clbits,
                seed_sequences,
                max_workers,
                engine_factory=DensityMatrixEngine,
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
        if request.density_matrix:
            state = self.export_state()

        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
            state=state,
        )

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("engine not initialized; call initialize(dims) first")


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None,
    clbits: list[int],
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    return condition is None or all(clbits[c] == v for c, v in condition)


def _execute_dynamic_plan_one_shot(
    engine: DensityMatrixEngine,
    plan: list[ResolvedStep],
    n_clbits: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Run one dynamic-path shot and return its final clbit snapshot.

    Serial-path twin of `statevector_engine._execute_dynamic_plan_one_shot`,
    which the parallel path reuses via `parallel.py`'s `engine_factory`
    dispatch; the two must stay behaviorally in sync (the equality is pinned
    by the parallel-vs-serial backend test). The only textual difference is
    that reset takes no rng here: it consumes no draws on this engine, so
    per-shot rng streams are not draw-for-draw aligned with the statevector
    engine's - counts for a fixed seed are reproducible per backend, not
    across backends.
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
                engine.reset_subsystems(step.reset_indices)
    return tuple(clbits)


def _analyze_plan_for_run(plan: list[ResolvedStep]) -> tuple[bool, list[tuple[int, int]]]:
    """Return this engine's dynamic-path decision and fast-path measurement pairs.

    Differs from the statevector analysis in one way: an unconditional
    ``ResetStep`` on never-measured subsystems does not force the dynamic
    path, because density-matrix reset is a deterministic channel the fast
    path applies inline. A reset still goes dynamic when it is conditioned or
    when it touches an already-measured subsystem (end-of-run sampling for
    that subsystem would otherwise read the post-reset state).
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
            if step.condition is not None:
                is_dynamic = True
            if any(t in measured_subsystems for t in step.reset_indices):
                is_dynamic = True
            continue
        if isinstance(step, ApplyMatrixStep):
            if step.condition is not None:
                is_dynamic = True
            if any(t in measured_subsystems for t in step.target_indices):
                is_dynamic = True
    return is_dynamic, measurements


def _requires_dynamic_execution(plan: list[ResolvedStep]) -> bool:
    """Return whether this engine must use per-shot execution for the plan."""
    is_dynamic, _measurements = _analyze_plan_for_run(plan)
    return is_dynamic


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
    out axes are permuted back into their positions, mirroring the statevector
    engine's ``_apply_matrix`` axis bookkeeping.
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


def _apply_matrix_rho(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply ``rho' = U_T rho U_T^dagger`` for a local matrix on flat ``targets``.

    The full ``prod(dims)``-dimensional ``U_T`` is never materialized: ``rho``
    is viewed as a ``2n``-axis ket/bra tensor and ``U`` (resp. ``conj(U)``) is
    contracted into the target ket (resp. bra) axes, matching the recommended
    dense reference path in ``matrix-engine.md`` §5.1.

    Complexity: two tensor contractions of O(prod(local_dims) * prod(dims)^2)
    FLOPs each, with ~2x-state peak memory per contraction (tensordot
    allocates, transpose may copy).
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


def _collapse_density_matrix(
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
    idx = int(rng.choice(size, p=_probabilities_from_rho(rho)))
    subsystems = list(measured_subsystems)

    if len(set(subsystems)) == len(dims):
        keep = np.zeros(size, dtype=bool)
        keep[idx] = True
    else:
        strides = _strides(dims)
        basis = np.arange(size)
        stride_arr = np.array([strides[q] for q in subsystems])
        dim_arr = np.array([dims[q] for q in subsystems])
        digits = (basis[:, None] // stride_arr) % dim_arr
        idx_digits = (idx // stride_arr) % dim_arr
        keep = np.all(digits == idx_digits, axis=1)

    new = rho * keep[:, None] * keep[None, :]
    trace = np.real(np.trace(new))
    if trace > 0:
        new = new / trace
    return idx, new


def _reset_density_matrix(
    rho: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Deterministically reset flat ``targets``: partial trace, then reprepare.

    Reference update from ``density-matrix-backend.md`` §5:
    ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)``. Working view is the
    ``(dt, dt, R, R)`` block structure of ``matrix-engine.md`` §5.2; the
    discard step traces the target ket/bra pair and the reprepare step writes
    the reduced state into the all-zero target block. Trace-preserving, so no
    renormalization is needed.
    """
    n = len(dims)
    k = len(targets)
    if reversed_dims is None:
        reversed_dims = tuple(dims[n - 1 - p] for p in range(n))
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

    rest_shape = tuple(
        (tuple(reversed_dims) * 2)[ax] for ax in remaining
    )
    post = post.reshape(local_dims + local_dims + rest_shape)
    inverse_perm = np.argsort(moved + remaining)
    return np.transpose(post, inverse_perm).reshape(rho.shape)


def _probabilities_from_rho(rho: np.ndarray) -> np.ndarray:
    """Return normalized computational-basis probabilities for a density matrix.

    The diagonal of a valid ``rho`` is real and non-negative; tiny negative
    round-off is clipped so the values remain a valid sampling distribution.
    """
    probabilities = np.clip(np.real(np.diagonal(rho)), 0.0, None)
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


