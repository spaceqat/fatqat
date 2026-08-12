"""NumPy engines for the matrix backend family.

`NumpySVEngine` (statevector) and `NumpyDMEngine` (density matrix) are the
two *state* `MatrixEngine` implementations. They share every semantics-agnostic
piece - strategy selection, the fast single-evolution path, the per-shot dynamic
path (serial or parallel across workers), ``initialize`` and
``measure_subsystems`` - through `_NumpyMatrixEngine`. Each leaf class then
contributes only its numeric kernels (allocate / apply / probabilities /
collapse / reset) plus two class knobs (``_state_field``,
``_reset_forces_dynamic``); no public method branches on semantics.

`NumpyUnitaryEngine` and `NumpySuperopEngine` are the two *operator* engines:
they compute the program's map rather than a state under it. Each is its state
twin evolved on many columns at once - a unitary is ``size`` statevector
columns (column ``j`` is ``U|j>``), a super-operator is ``size**2``
density-matrix columns (column ``b`` is the image of basis matrix ``E_b``).
`_NumpyOperatorEngine` replaces ``run`` with one deterministic pass.

Conventions:

- little-endian: the flat basis-index digit for subsystem ``q`` has place value
  ``prod(dims[:q])``; subsystem 0 is the least-significant digit.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
- a density matrix reshaped to ``reversed(dims) + reversed(dims)`` exposes ket
  axes ``[0, n)`` and bra axes ``[n, 2n)``; subsystem ``q``'s ket axis is
  ``n - 1 - q`` and its bra axis is ``2n - 1 - q``.

Semantics differences:

- statevector evolution is ``|psi'> = U_T |psi>``; density-matrix evolution is
  ``rho' = U_T rho U_T^dagger``. Neither materializes the global ``U_T``.
- measurement is trajectory-style on both: Born sampling, projective collapse,
  normalization.
- reset samples a branch on the statevector (one rng draw), so any reset forces
  the per-shot path; on the density matrix it is the deterministic
  partial-trace channel (no rng draw), so an unconditional reset stays on the
  fast path. Per-shot rng streams are therefore not draw-for-draw aligned across
  the two, so counts for a fixed seed are reproducible per engine, not across
  engines.
- a channel (``ApplyChannelStep``) follows the same split as reset: the
  statevector samples one Kraus branch per occurrence (quantum-jump
  unravelling, one rng draw), forcing the per-shot path; the density matrix
  applies the exact Kraus sum ``sum_i K_i rho K_i^dagger`` (no rng draw), so
  an unconditional channel stays on the fast path.
- measurement reporting is identical on both: the collapse keeps the physical
  outcome, maps it to a reported digit, then optionally resamples that digit
  through ``MeasurementStep.confusions``. It never affects path
  classification. On the per-shot path feedforward conditions read the
  (possibly confused) reported value.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import prod

import numpy as np

from ..._backends._execution_analysis import (
    _OperationExecutionFacts,
    _analyze_terminal_measurements,
)
from ..._backends.engine_contract import _EngineConfig as EngineConfig, RawResult
from ..._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    AtomLossStep,
    OccupancyInitStep,
    MeasurementStep,
    ResetStep,
    RefillStep,
    ResolvedStep,
)
from ...implementation.matrices import shift_matrix
from ...result import decode_indices_to_clbit_rows, reduce_to_counts
from .base import ResultRequest, MatrixEngine

ERASURE_DIGIT = 2

# --- shared numeric primitives ---


def _digit(flat: int, index: int, dims: Sequence[int]) -> int:
    """Little-endian digit of subsystem ``index`` in flat basis state ``flat``."""
    stride = 1
    for q in range(index):
        stride *= dims[q]
    return (flat // stride) % dims[index]


def _strides(dims: Sequence[int]) -> list[int]:
    """Little-endian place values: ``stride[q] = prod(dims[:q])``."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def _contract_local(
    m: np.ndarray, tensor: np.ndarray, axes: Sequence[int], total: int, k: int
) -> np.ndarray:
    """Contract a local ``[out, in]`` operator into ``axes`` of a tensor.

    ``m`` has ``2k`` axes (k out, then k in); ``tensor`` has ``total`` axes. The
    k input axes of ``m`` are contracted with ``axes`` and the resulting out axes
    are permuted back into their positions. This is the single matrix-application
    core both apply kernels share: the statevector kernel calls it once on the
    n-axis state tensor, the density-matrix kernel twice (ket then bra) on the
    2n-axis rho tensor.
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
    idx: int, subsystems: Sequence[int], dims: Sequence[int], size: int
) -> np.ndarray:
    """Boolean mask over flat basis states matching ``idx``'s measured digits.

    The projector core both collapse kernels share: an entry is kept when every
    measured subsystem's little-endian digit equals the corresponding digit of
    the sampled index ``idx``.
    """
    subsystems = list(subsystems)
    if len(set(subsystems)) == len(dims):
        keep = np.zeros(size, dtype=bool)
        keep[idx] = True
        return keep
    strides = _strides(dims)
    basis = np.arange(size)
    # One (N, m) broadcast instead of m separate O(N) passes: fold every measured
    # subsystem's stride and modulus into a single vectorized divide/mod/compare.
    stride_arr = np.array([strides[q] for q in subsystems])
    dim_arr = np.array([dims[q] for q in subsystems])
    digits = (basis[:, None] // stride_arr) % dim_arr
    idx_digits = (idx // stride_arr) % dim_arr
    return np.all(digits == idx_digits, axis=1)


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, clbits: list[int]
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    return condition is None or all(clbits[c] == v for c, v in condition)


def _map_physical_digit(physical_digit: int, reported_digit_map) -> int:
    """Map one physical measurement outcome to its reported classical digit."""
    if reported_digit_map is None:
        return physical_digit
    return reported_digit_map[physical_digit]


def _report_digit(reported_digit: int, confusion, rng: np.random.Generator) -> int:
    """Optionally resample one reported digit through readout error.

    ``confusion`` is column-stochastic (``C[i, j] = P(report i | mapped j)``)
    or ``None`` for an error-free readout. Only the reported classical value
    is affected; the caller's collapsed state keeps the physical outcome.
    """
    if confusion is None:
        return reported_digit
    return int(rng.choice(confusion.shape[0], p=confusion[:, reported_digit]))


def _reporting_by_clbit(
    plan: list[ResolvedStep],
) -> dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]]:
    """Map each clbit to the reporting contract of its last writer.

    Fast-path counts decode only each clbit's final value (later measurement
    writes replace earlier ones, see ``decode_indices_to_clbit_rows``), so the
    physical-to-reported map and its readout confusion must likewise come from
    the last writer. An earlier map or confusion must not survive an error-free
    overwrite.
    """
    by_clbit: dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]] = {}
    for step in plan:
        if isinstance(step, MeasurementStep):
            confusions = step.confusions or (None,) * len(step.classical_indices)
            maps = step.reported_digit_maps or (None,) * len(step.classical_indices)
            for clbit, reported_map, confusion in zip(
                step.classical_indices, maps, confusions
            ):
                by_clbit[clbit] = (reported_map, confusion)
    return by_clbit


def _apply_measurement_reporting(
    rows: np.ndarray,
    by_clbit: dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]],
    rng: np.random.Generator,
) -> None:
    """Map physical decoded columns, then apply readout confusion in place."""
    for clbit, (reported_map, confusion) in by_clbit.items():
        if reported_map is not None:
            rows[:, clbit] = np.asarray(reported_map, dtype=int)[rows[:, clbit]]
        if confusion is None:
            continue
        dim = confusion.shape[0]
        reported_column = rows[:, clbit].copy()
        for reported_digit in range(dim):
            mask = reported_column == reported_digit
            hits = int(mask.sum())
            if hits:
                rows[mask, clbit] = rng.choice(
                    dim, size=hits, p=confusion[:, reported_digit]
                )


# --- shared orchestration ---


class _NumpyMatrixEngine(MatrixEngine):
    """Semantics-agnostic execution for the NumPy matrix-family engines.

    Owns strategy selection and both run paths and expresses them purely through
    the abstract kernels. Subclasses supply ``_allocate``, ``apply``,
    ``apply_channel``, ``probabilities``, ``collapse``, ``reset_subsystems``
    and two class knobs:

    - ``_state_field``: the request/result state field this engine populates
      (``"statevector"`` or ``"density_matrix"``).
    - ``_reset_forces_dynamic``: whether a reset makes execution stochastic and
      thus forces the per-shot path (statevector: True; density matrix: False).
    """

    _state_field: str
    _reset_forces_dynamic: bool

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        self._set_dims(system_dims)
        self._n_clbits = int(n_clbits)
        self._state = self._allocate(prod(self._dims) if self._dims else 1)

    @abstractmethod
    def _allocate(self, size: int) -> np.ndarray:
        """Return the all-zero computational state over ``size`` basis states."""

    @abstractmethod
    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply a Kraus channel to the internal state in place.

        Consumes ``rng`` only under statevector semantics (one draw to sample
        the trajectory branch); the density-matrix kernel is deterministic
        and accepts it for interface parity, like reset.
        """

    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]:
        assert len(indices) >= 1, "measure_subsystems requires at least one index"
        flat = self.collapse(indices, rng)
        return tuple(_digit(flat, index, self._dims) for index in indices)

    def run(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: ResultRequest,
        *,
        config: EngineConfig | None = None,
    ) -> RawResult:
        assert (
            self._state is not None
        ), "engine not initialized; call initialize() first"
        is_dynamic, measurements = self._analyze_plan(plan)
        if is_dynamic:
            return self._run_per_shot(plan, shots, seed, request, config or self.config)
        return self._run_fast(
            plan, measurements, shots, np.random.default_rng(seed), request
        )

    def _analyze_plan(
        self, plan: list[ResolvedStep]
    ) -> tuple[bool, list[tuple[int, int]]]:
        """Return the shared fast-path decision and matrix measurement pairs."""
        is_dynamic, measurements = _analyze_terminal_measurements(
            plan, self._operation_execution_facts
        )
        return is_dynamic, [
            pair
            for step in measurements
            for pair in zip(step.measured_indices, step.classical_indices)
        ]

    def _operation_execution_facts(
        self, step: ResolvedStep
    ) -> _OperationExecutionFacts:
        """Describe one matrix operation for shared dynamic-plan analysis."""
        if isinstance(step, ResetStep):
            return _OperationExecutionFacts(
                target_indices=step.reset_indices,
                is_conditioned=step.condition is not None,
                forces_per_shot=self._reset_forces_dynamic,
            )
        if isinstance(step, ApplyChannelStep):
            return _OperationExecutionFacts(
                target_indices=step.target_indices,
                is_conditioned=step.condition is not None,
                forces_per_shot=self._reset_forces_dynamic,
            )
        if isinstance(step, ApplyMatrixStep):
            return _OperationExecutionFacts(
                target_indices=step.target_indices,
                is_conditioned=step.condition is not None,
            )
        if isinstance(step, AtomLossStep):
            return _OperationExecutionFacts(
                target_indices=step.target_indices,
                is_conditioned=step.condition is not None,
                forces_per_shot=True,   
            )
        if isinstance(step, RefillStep):
            return _OperationExecutionFacts(
                target_indices=step.target_indices,
                is_conditioned=step.condition is not None,
                forces_per_shot=True,
            )
        if isinstance(step, OccupancyInitStep):
            return _OperationExecutionFacts(
                target_indices=(),
                is_conditioned=False,
                forces_per_shot=False,
            )
        raise TypeError(f"unknown resolved execution step {type(step).__name__}")

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        measurements: list[tuple[int, int]],
        shots: int,
        rng: np.random.Generator,
        request: ResultRequest,
    ) -> RawResult:
        """Evolve once, optionally sample counts, optionally export the state.

        The ``ResetStep`` and ``ApplyChannelStep`` branches are only reachable
        under density-matrix semantics, where both are deterministic channels;
        statevector semantics routes every plan bearing them to the dynamic
        path.
        """
        state_requested = getattr(request, self._state_field)
        self.initialize(self._dims, self._n_clbits)
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                self.apply(step)
            elif isinstance(step, ApplyChannelStep):
                self.apply_channel(step, rng)
            elif isinstance(step, ResetStep):
                self.reset_subsystems(step.reset_indices, rng)

        collapsed_index: int | None = None
        if state_requested and measurements:
            collapsed_index = self.collapse([q for q, _ in measurements], rng)

        outcome_keys = outcome_counts = state = None
        if request.counts:
            if not measurements:
                indices = np.zeros(shots, dtype=int)
            elif collapsed_index is not None:
                indices = np.array([collapsed_index], dtype=int)
            else:
                indices = self.sample_indices(shots, rng)
            rows = decode_indices_to_clbit_rows(
                indices, measurements, self._dims, self._n_clbits
            )
            _apply_measurement_reporting(rows, _reporting_by_clbit(plan), rng)
            outcome_keys, outcome_counts = reduce_to_counts(rows)
        if state_requested:
            state = self.export_state()
        return RawResult(
            outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=state
        )

    def _run_per_shot(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: ResultRequest,
        config: EngineConfig,
    ) -> RawResult:
        """Run dynamic execution one trajectory at a time or via worker batches."""
        from .parallel import (
            _planned_workers,
            _run_dynamic_shots_parallel,
            _shot_seed_sequences,
        )

        state_requested = getattr(request, self._state_field)
        n_iters = shots if request.counts else (1 if state_requested else 0)
        seed_sequences = _shot_seed_sequences(seed, n_iters)

        # A state export must come from this process, so it never parallelizes.
        max_workers = (
            None if state_requested else _planned_workers(config, request, n_iters)
        )
        if max_workers is not None:
            snapshots = _run_dynamic_shots_parallel(
                config,
                plan,
                self._dims,
                self._n_clbits,
                seed_sequences,
                max_workers,
                type(self),
            )
        else:
            snapshots = []
            for seed_sequence in seed_sequences:
                self.initialize(self._dims, self._n_clbits)
                snapshots.append(
                    self._run_one_shot(plan, np.random.default_rng(seed_sequence))
                )

        outcome_keys = outcome_counts = state = None
        if request.counts:
            rows = np.asarray(snapshots, dtype=int).reshape(
                (len(snapshots), self._n_clbits)
            )
            outcome_keys, outcome_counts = reduce_to_counts(rows)
        if state_requested:
            state = self.export_state()
        return RawResult(
            outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=state
        )

    def _run_one_shot(
        self, plan: list[ResolvedStep], rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Run one dynamic-path shot and return its final clbit snapshot.

        Shared by the serial path and `parallel.py`'s worker batches; it touches
        the state only through the interface methods (reset consumes rng only
        under statevector semantics).
        """
        clbits = [0] * self._n_clbits
        occupied = set(range(len(self._dims)))
        for step in plan:
            if isinstance(step, OccupancyInitStep):
                occupied = set(step.occupied_indices)  
                continue
            if isinstance(step, ApplyMatrixStep) and all(
                    t in occupied for t in step.target_indices
                ):
                if _condition_matches(step.condition, clbits):
                    self.apply(step)
            elif isinstance(step, ApplyChannelStep) and all(
                    t in occupied for t in step.target_indices
                ):
                if _condition_matches(step.condition, clbits):
                    self.apply_channel(step, rng)
            elif isinstance(step, AtomLossStep):
                if _condition_matches(step.condition, clbits):
                    for index in step.target_indices:
                        if index in occupied and rng.random() < step.p:
                            occupied.discard(index)
                            self.reset_subsystems([index], rng)
            elif isinstance(step, RefillStep):
                if _condition_matches(step.condition, clbits):
                    for index in step.target_indices:
                        if index not in occupied:
                            occupied.add(index)
                            self.reset_subsystems([index], rng)
            elif isinstance(step, MeasurementStep):
                bits = self.measure_subsystems(step.measured_indices, rng)
                confusions = step.confusions or (None,) * len(bits)
                maps = step.reported_digit_maps or (None,) * len(bits)
                # The collapse keeps the physical outcome; only its mapped,
                # optionally confused report is written to the clbits.
                for m, c, bit, reported_map, confusion in zip(
                    step.measured_indices, step.classical_indices, bits, maps, confusions
                ):
                    if m not in occupied:
                        clbits[c] = ERASURE_DIGIT
                    else:
                        clbits[c] = _report_digit(
                            _map_physical_digit(bit, reported_map), confusion, rng
                        )
            elif isinstance(step, ResetStep):
                if _condition_matches(step.condition, clbits) and all(
                    t in occupied for t in step.reset_indices
                ):
                    self.reset_subsystems(step.reset_indices, rng)
        return tuple(clbits)


# --- statevector engine ---


class NumpySVEngine(_NumpyMatrixEngine):
    """State-vector engine: evolves ``|psi>`` as a flat little-endian array."""

    _state_field = "statevector"
    _reset_forces_dynamic = True

    def __init__(self, name: str = "numpy-sv", config: EngineConfig | None = None):
        super().__init__(name, config, state_semantics="sv")

    def _allocate(self, size: int) -> np.ndarray:
        state = np.zeros(size, dtype=complex)
        state[0] = 1.0
        return state

    def apply(self, step: ApplyMatrixStep) -> None:
        self._state = self._apply_local(self.state, step.matrix, step.target_indices)

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Sample one Kraus branch (quantum-jump unravelling) and renormalize.

        Each candidate branch ``K_i |psi>`` is produced by the same local-apply
        primitive gates use; its squared norm is the branch probability. One
        branch is drawn (consuming one rng draw, same posture as measurement
        and reset) and kept, normalized.

        Each branch starts from a fresh copy of the state: ``_apply_local`` is
        only contracted to *return* the new state, and a subclass kernel (the
        Numba one) legitimately updates its input buffer in place - reusing
        ``self.state`` across branches would then corrupt the source mid-loop.
        """
        source = self.state
        branches = [
            self._apply_local(source.copy(), kraus, step.target_indices)
            for kraus in step.kraus_ops
        ]
        norms = np.array([np.real(np.vdot(b, b)) for b in branches])
        # CPTP guarantees the norms sum to <psi|psi> = 1; renormalize anyway so
        # rng.choice never rejects the distribution over float round-off.
        chosen = int(rng.choice(len(branches), p=norms / norms.sum()))
        self._state = branches[chosen] / np.sqrt(norms[chosen])

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a local matrix to flat ``targets``, returning the new flat state."""
        n = len(self._dims)
        k = len(targets)
        local_dims = tuple(self._dims[t] for t in targets)
        psi = state.reshape(self._reversed_dims)
        target_axes = [n - 1 - q for q in targets]
        m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
        return _contract_local(m, psi, target_axes, n, k).reshape(-1)

    def probabilities(self) -> np.ndarray:
        probabilities = np.abs(self.state) ** 2
        total = probabilities.sum()
        return probabilities / total if total > 0 else probabilities

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        state = self.state
        idx = int(rng.choice(len(state), p=self.probabilities()))
        keep = _measured_keep_mask(idx, measured_subsystems, self._dims, len(state))
        new = state.copy()
        new[~keep] = 0.0
        norm = np.linalg.norm(new)
        self._state = new / norm if norm > 0 else new
        return idx

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None:
        """Measure ``indices`` as one branch, then shift each nonzero digit to ``|0>``.

        Consumes exactly one rng draw; the rest of an entangled state is left
        correctly conditioned on the sampled branch.
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        idx = self.collapse(indices, rng)
        for index in indices:
            outcome = _digit(idx, index, self._dims)
            if outcome != 0:
                inv = shift_matrix(self._dims[index], -outcome)
                self._state = self._apply_local(self._state, inv, (index,))


# --- density-matrix engine ---


class NumpyDMEngine(_NumpyMatrixEngine):
    """Density-matrix engine: evolves ``rho`` as a ``(size, size)`` matrix."""

    _state_field = "density_matrix"
    _reset_forces_dynamic = False

    def __init__(self, name: str = "numpy-dm", config: EngineConfig | None = None):
        super().__init__(name, config, state_semantics="dm")

    def _allocate(self, size: int) -> np.ndarray:
        state = np.zeros((size, size), dtype=complex)
        state[0, 0] = 1.0
        return state

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply ``rho' = U_T rho U_T^dagger`` without materializing ``U_T``."""
        self._state = self._apply_local_sandwich(
            self.state, step.matrix, step.target_indices
        )

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply the exact Kraus sum ``rho' = sum_i K_i rho K_i^dagger``.

        Each term reuses the two-sided local-apply primitive gates use; the
        branch weights are implicit in the ``K_i rho K_i^dagger`` sandwiches,
        which is what makes this exact. No randomness is consumed (``rng`` is
        accepted only for interface parity, like reset).
        """
        rho = self.state
        new = np.zeros_like(rho)
        for kraus in step.kraus_ops:
            new += self._apply_local_sandwich(rho, kraus, step.target_indices)
        self._state = new

    def _apply_local_sandwich(
        self, rho: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Return ``M_T rho M_T^dagger`` for a local matrix on flat ``targets``.

        ``rho`` is viewed as a 2n-axis ket/bra tensor and ``M`` (resp.
        ``conj(M)``) is contracted into the target ket (resp. bra) axes. The
        matrix need not be unitary: gates and Kraus operators share this
        primitive, differing only in whether the caller sums over terms.
        """
        n = len(self._dims)
        k = len(targets)
        local_dims = tuple(self._dims[t] for t in targets)
        tensor = rho.reshape(self._reversed_dims * 2)
        m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
        ket_axes = [n - 1 - q for q in targets]
        bra_axes = [2 * n - 1 - q for q in targets]
        tensor = _contract_local(m, tensor, ket_axes, 2 * n, k)
        tensor = _contract_local(m.conj(), tensor, bra_axes, 2 * n, k)
        return tensor.reshape(rho.shape)

    def probabilities(self) -> np.ndarray:
        # A valid rho's diagonal is real and non-negative; clip tiny round-off so
        # the values stay a valid sampling distribution.
        probabilities = np.clip(np.real(np.diagonal(self.state)), 0.0, None)
        total = probabilities.sum()
        return probabilities / total if total > 0 else probabilities

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        rho = self.state
        size = rho.shape[0]
        idx = int(rng.choice(size, p=self.probabilities()))
        keep = _measured_keep_mask(idx, measured_subsystems, self._dims, size)
        new = rho * keep[:, None] * keep[None, :]
        trace = np.real(np.trace(new))
        self._state = new / trace if trace > 0 else new
        return idx

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Deterministically reset ``indices``: partial trace, then reprepare ``|0>``.

        ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)``. Trace-preserving, so
        no renormalization is needed and no randomness is consumed (``rng`` is
        accepted only for interface parity).
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        rho = self.state
        n = len(self._dims)
        targets = tuple(indices)
        local_dims = tuple(self._dims[t] for t in targets)
        dt = prod(local_dims)
        rest = rho.shape[0] // dt

        tensor = rho.reshape(self._reversed_dims * 2)
        ket_axes = [n - 1 - q for q in targets]
        bra_axes = [2 * n - 1 - q for q in targets]
        moved = ket_axes + bra_axes
        # `remaining` stays ascending, so rest-ket axes precede rest-bra axes and
        # the (dt, dt, rest, rest) regroup below is valid.
        remaining = [ax for ax in range(2 * n) if ax not in moved]
        block = np.transpose(tensor, moved + remaining).reshape(dt, dt, rest, rest)

        rho_rest = np.trace(block, axis1=0, axis2=1)
        post = np.zeros_like(block)
        post[0, 0] = rho_rest

        rest_shape = tuple((self._reversed_dims * 2)[ax] for ax in remaining)
        post = post.reshape(local_dims + local_dims + rest_shape)
        inverse_perm = np.argsort(moved + remaining)
        self._state = np.transpose(post, inverse_perm).reshape(rho.shape)


# --- operator engines ---


class _NumpyOperatorEngine(_NumpyMatrixEngine):
    """Deterministic single-pass execution for the operator representations.

    Replaces `_NumpyMatrixEngine`'s fast/per-shot split with one evolution of
    the identity operator; the sampling kernels are unsupported.
    """

    def run(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: ResultRequest,
        *,
        config: EngineConfig | None = None,
    ) -> RawResult:
        """Evolve the identity operator once through ``plan`` and export it.

        ``shots``, ``seed``, and ``config`` are accepted for interface parity
        and unused, as is the ``rng`` handed to the exact-channel kernels.
        """
        assert (
            self._state is not None
        ), "engine not initialized; call initialize() first"
        self.initialize(self._dims, self._n_clbits)
        rng = np.random.default_rng(seed)
        for step in plan:
            assert not isinstance(
                step, MeasurementStep
            ), "operator execution cannot represent a measurement"
            assert (
                step.condition is None
            ), "operator execution cannot represent a feedforward condition"
            if isinstance(step, ApplyMatrixStep):
                self.apply(step)
            elif isinstance(step, ApplyChannelStep):
                self.apply_channel(step, rng)
            elif isinstance(step, ResetStep):
                self.reset_subsystems(step.reset_indices, rng)
            else:
                raise TypeError(
                    f"unknown resolved execution step {type(step).__name__}"
                )

        state = self.export_state() if getattr(request, self._state_field) else None
        return RawResult(state=state)

    def probabilities(self) -> np.ndarray:
        """Unsupported: an operator is a map, not a distribution over states."""
        raise NotImplementedError(
            f"{self._state_field} execution has no basis-state distribution"
        )

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Unsupported: an operator cannot be projected onto one outcome."""
        raise NotImplementedError(
            f"{self._state_field} execution cannot represent a measurement"
        )


class NumpyUnitaryEngine(_NumpyOperatorEngine, NumpySVEngine):
    """Unitary engine: evolves ``U`` as a ``(size, size)`` operator matrix.

    Column ``j`` of ``U`` is the statevector ``U|j>``, so the whole operator is
    the statevector kernel run on ``size`` columns at once.
    """

    _state_field = "unitary"

    def __init__(self, name: str = "numpy-unitary", config: EngineConfig | None = None):
        super().__init__(name, config)

    def _allocate(self, size: int) -> np.ndarray:
        return np.eye(size, dtype=complex)

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a local matrix to every column of the operator at once."""
        n = len(self._dims)
        k = len(targets)
        local_dims = tuple(self._dims[t] for t in targets)
        columns = state.shape[1]
        tensor = state.reshape(self._reversed_dims + (columns,))
        m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
        target_axes = [n - 1 - q for q in targets]
        return _contract_local(m, tensor, target_axes, n + 1, k).reshape(state.shape)


class NumpySuperopEngine(_NumpyOperatorEngine, NumpyDMEngine):
    """Super-operator engine: evolves ``S`` as a ``(size**2, size**2)`` matrix.

    Column ``b`` of ``S`` is the vectorized image of basis matrix ``E_b``, so
    the whole channel is the density-matrix kernel run on ``size**2`` columns at
    once, in the row-major vectorization the density matrix already uses
    (``vec(rho) = rho.reshape(-1)``, i.e. ``S = kron(M, conj(M))``).
    """

    _state_field = "superop"

    def __init__(self, name: str = "numpy-superop", config: EngineConfig | None = None):
        super().__init__(name, config)
        # Keyed by (subsystem, dimension), so an entry can never go stale.
        self._reset_channels: dict[tuple[int, int], ApplyChannelStep] = {}

    def _allocate(self, size: int) -> np.ndarray:
        return np.eye(size * size, dtype=complex)

    def _apply_local_sandwich(
        self, rho: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Return ``M_T . M_T^dagger`` applied to every column at once."""
        n = len(self._dims)
        k = len(targets)
        local_dims = tuple(self._dims[t] for t in targets)
        columns = rho.shape[1]
        tensor = rho.reshape(self._reversed_dims * 2 + (columns,))
        m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
        total = 2 * n + 1
        ket_axes = [n - 1 - q for q in targets]
        bra_axes = [2 * n - 1 - q for q in targets]
        tensor = _contract_local(m, tensor, ket_axes, total, k)
        tensor = _contract_local(m.conj(), tensor, bra_axes, total, k)
        return tensor.reshape(rho.shape)

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Reset ``indices`` through the Kraus channel ``sum_k |0><k| . |k><0|``.

        Equal to the density matrix's partial-trace reset. Deterministic -
        ``rng`` is accepted only for interface parity.
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        for index in indices:
            self.apply_channel(self._reset_channel(index), rng)

    def _reset_channel(self, index: int) -> ApplyChannelStep:
        """The single-subsystem reset channel for ``index``, built once."""
        dim = self._dims[index]
        step = self._reset_channels.get((index, dim))
        if step is None:
            kraus = []
            for outcome in range(dim):
                operator = np.zeros((dim, dim), dtype=complex)
                operator[0, outcome] = 1.0
                kraus.append(operator)
            step = ApplyChannelStep(kraus_ops=tuple(kraus), target_indices=(index,))
            self._reset_channels[(index, dim)] = step
        return step
