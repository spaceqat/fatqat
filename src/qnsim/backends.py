"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import importlib.util
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Any

import numpy as np

from .engine import StateVectorEngine
from .errors import BackendValidationError, NoMeasurementWarning, UnsupportedOperationError
from .implementation import ApplyMatrixStep, default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .operations import ResetGate
from .program import AppliedOperation, Measurement, Program
from .result import (
    Result,
    ResultConfig,
    build_counts,
    build_counts_from_clbits,
    count_key_from_clbits,
)


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat qubit indices into matching flat clbit indices."""

    measured_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.measured_indices) != len(self.classical_indices):
            raise ValueError("measurement step indices must have equal length")
        if len(self.measured_indices) < 1:
            raise ValueError("measurement step requires at least one index")


@dataclass(frozen=True)
class ResetStep:
    """Resolved reset of one or more flat qubits to |0>, with optional condition.

    `Reset` is an `AppliedOperation`, so it can carry a feedforward `condition`
    just like a gate. The lowered form stores it as ``(clbit_index, value)``
    AND-terms; the per-shot loop skips the reset when the guard fails.
    """

    reset_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.reset_indices) < 1:
            raise ValueError("reset step requires at least one index")


ResolvedStep = ApplyMatrixStep | MeasurementStep | ResetStep


@dataclass(frozen=True)
class _PlanFacts:
    """Classification facts computed while lowering a program."""

    is_dynamic: bool
    has_measurement: bool
    has_reset: bool


@dataclass(frozen=True)
class _ResultRequest:
    """Resolved result fields requested for one execution."""

    counts: bool
    statevector: bool


@dataclass(frozen=True)
class _BackendConfig:
    """Normalized backend execution-strategy options."""

    max_workers: Any = None
    parallel_backend: Any = "auto"


def _normalize_backend_options(options: dict[str, Any] | None) -> _BackendConfig:
    if options is None:
        return _BackendConfig()
    known = {"max_workers", "parallel_backend"}
    ignored = {key: value for key, value in options.items() if key not in known}
    if ignored:
        warnings.warn(
            f"StateVectorBackend ignored unsupported backend options: {ignored!r}",
            UserWarning,
            stacklevel=3,
        )
    return _BackendConfig(
        max_workers=options.get("max_workers"),
        parallel_backend=options.get("parallel_backend", "auto"),
    )


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    layout: ResourceLayout,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    return tuple((layout.clbit_index(ref), int(val)) for ref, val in condition)


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None,
    clbits: list[int],
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    return condition is None or all(clbits[c] == v for c, v in condition)


def _shot_seed_sequences(seed: int | None, n_iters: int) -> list[np.random.SeedSequence]:
    """Spawn one independent child `SeedSequence` per logical shot.

    Child streams are derived from a single root sequence in shot order, so
    serial and (future) parallel execution draw from the same reproducible
    per-shot streams regardless of how shots are distributed across workers.
    """
    root = np.random.SeedSequence(seed)
    return root.spawn(n_iters)


def _execute_dynamic_plan_one_shot(
    engine: StateVectorEngine,
    plan: list[ResolvedStep],
    n_clbits: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Run one dynamic-path shot to completion and return its clbit snapshot."""
    clbits = [0] * n_clbits
    for step in plan:
        if isinstance(step, ApplyMatrixStep):
            if _condition_matches(step.condition, clbits):
                engine.apply(step)
        elif isinstance(step, MeasurementStep):
            bits = engine.measure_qubits(step.measured_indices, rng)
            for c, bit in zip(step.classical_indices, bits):
                clbits[c] = bit
        else:  # ResetStep
            if _condition_matches(step.condition, clbits):
                engine.reset_qubits(step.reset_indices, rng)
    return tuple(clbits)


def _run_dynamic_shot(
    plan: list[ResolvedStep],
    n_qubits: int,
    n_clbits: int,
    seed_sequence: np.random.SeedSequence,
) -> tuple[int, ...]:
    """Pickle-safe, top-level single-shot worker for parallel dynamic execution.

    Builds its own engine and RNG from its own child seed sequence, so it has
    no shared mutable state with any other shot.
    """
    rng = np.random.default_rng(seed_sequence)
    engine = StateVectorEngine()
    engine.initialize(n_qubits)
    return _execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng)


def _loky_available() -> bool:
    return importlib.util.find_spec("loky") is not None


# Minimum shot count before automatic parallelism (max_workers=None) kicks in.
# Below this, process-pool startup and per-batch pickling cost more than the
# per-shot simulation they save, so automatic mode stays serial — this keeps the
# default backend and small runs (including the existing Phase 2 dynamic tests)
# fast instead of spawning workers for a handful of shots. The value is a tunable
# heuristic (it keys off shot count, not per-shot cost); 32 is chosen to stay
# serial for typical small/interactive runs while parallelizing real multishot
# jobs. An explicit max_workers > 1 always parallelizes and bypasses this floor.
_PARALLEL_MIN_SHOTS = 32


def _effective_max_workers(max_workers: object, n_iters: int) -> int:
    if max_workers is None:
        return min(n_iters, os.process_cpu_count() or 1)
    return int(max_workers)


def _should_parallelize(
    config: _BackendConfig,
    request: _ResultRequest,
    n_iters: int,
) -> bool:
    if not request.counts:
        return False
    if n_iters <= 1:
        return False
    if config.parallel_backend == "serial":
        return False
    if config.max_workers is None and n_iters < _PARALLEL_MIN_SHOTS:
        # Automatic mode stays serial for small runs; explicit max_workers wins.
        return False
    workers = _effective_max_workers(config.max_workers, n_iters)
    return workers > 1


def _resolve_parallel_backend_name(name: object) -> str:
    if name == "auto":
        return "loky" if _loky_available() else "multiprocessing"
    if name in {"serial", "multiprocessing", "loky"}:
        return str(name)
    raise BackendValidationError(f"unsupported parallel_backend={name!r}")


def _split_into_batches(
    seed_sequences: list[np.random.SeedSequence],
    n_batches: int,
) -> list[list[np.random.SeedSequence]]:
    """Split shots into up to n_batches contiguous batches, one task per worker.

    Batching pickles the (matrix-carrying) plan once per worker instead of once
    per shot, and lets each worker reuse a single engine across its batch. Shot i
    keeps ``seed_sequences[i]`` regardless of batching, so aggregated counts are
    byte-for-byte identical to the serial path.
    """
    n_items = len(seed_sequences)
    n_batches = max(1, min(n_batches, n_items))
    base, extra = divmod(n_items, n_batches)
    batches: list[list[np.random.SeedSequence]] = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        batches.append(seed_sequences[start : start + size])
        start += size
    return batches


def _run_dynamic_shot_batch(
    plan: list[ResolvedStep],
    n_qubits: int,
    n_clbits: int,
    seed_batch: list[np.random.SeedSequence],
) -> list[tuple[int, ...]]:
    """Run a contiguous batch of dynamic shots in one worker with one engine."""
    engine = StateVectorEngine()
    snapshots: list[tuple[int, ...]] = []
    for seed_sequence in seed_batch:
        rng = np.random.default_rng(seed_sequence)
        engine.initialize(n_qubits)
        snapshots.append(_execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng))
    return snapshots


def _run_dynamic_shots_multiprocessing(
    plan: list[ResolvedStep],
    n_qubits: int,
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    batches = _split_into_batches(seed_sequences, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _run_dynamic_shot_batch,
            repeat(plan),
            repeat(n_qubits),
            repeat(n_clbits),
            batches,
        )
        return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_loky(
    plan: list[ResolvedStep],
    n_qubits: int,
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    from loky import get_reusable_executor

    batches = _split_into_batches(seed_sequences, max_workers)
    executor = get_reusable_executor(max_workers=max_workers)
    results = executor.map(
        _run_dynamic_shot_batch,
        repeat(plan),
        repeat(n_qubits),
        repeat(n_clbits),
        batches,
    )
    return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_parallel(
    config: _BackendConfig,
    plan: list[ResolvedStep],
    n_qubits: int,
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
) -> list[tuple[int, ...]]:
    max_workers = min(
        _effective_max_workers(config.max_workers, len(seed_sequences)),
        len(seed_sequences),
    )
    backend_name = _resolve_parallel_backend_name(config.parallel_backend)
    if backend_name == "serial":
        return [_run_dynamic_shot(plan, n_qubits, n_clbits, ss) for ss in seed_sequences]
    if backend_name == "multiprocessing":
        return _run_dynamic_shots_multiprocessing(
            plan, n_qubits, n_clbits, seed_sequences, max_workers
        )
    if backend_name == "loky":
        if not _loky_available():
            warnings.warn(
                "parallel_backend='loky' requested but loky is unavailable; "
                "falling back to multiprocessing",
                UserWarning,
                stacklevel=3,
            )
            return _run_dynamic_shots_multiprocessing(
                plan, n_qubits, n_clbits, seed_sequences, max_workers
            )
        return _run_dynamic_shots_loky(
            plan, n_qubits, n_clbits, seed_sequences, max_workers
        )
    raise BackendValidationError(f"unsupported parallel_backend={backend_name!r}")


def _resolve_result_request(config: ResultConfig, facts: _PlanFacts) -> _ResultRequest:
    """Resolve default result fields from config and lowered program facts."""
    stochastic = facts.has_measurement or facts.has_reset
    counts = config.counts if config.counts is not None else facts.has_measurement
    statevector = config.statevector
    if statevector is None:
        statevector = not stochastic
    return _ResultRequest(counts=counts, statevector=statevector)


class StateVectorBackend:
    """Phase 1 statevector backend for qubit programs.

    The backend validates a `Program`, evolves supported operations with the
    matrix engine, samples terminal measurements, and returns an eager `Job`.
    A backend instance reuses one engine across runs, so it is suitable for
    repeated single-threaded use but not concurrent `run` calls.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Create a statevector backend."""
        self._config = _normalize_backend_options(options)
        self._impl_map = default_implementation_map()
        # The engine is constructed once and re-initialized per run so its
        # compiled kernels can be reused. Because it holds per-run state, a
        # single backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._engine = StateVectorEngine()

    def resolve_layout(self, program: Program) -> ResourceLayout:
        """Build the flat resource layout used by this backend.

        Args:
            program: Program whose registers should be flattened.

        Returns:
            Resource layout mapping register references to flat indices.
        """
        return ResourceLayout.from_program(program)

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        result_config: ResultConfig | None = None,
        seed: int | None = None,
    ) -> Job:
        """Validate and execute a program.

        Counts default to available when the program contains measurements.
        Statevector output defaults to available only when execution is
        deterministic (no measurement/reset sampling).

        Args:
            program: Program to execute.
            shots: Number of samples used when counts are requested.
            result_config: Optional `ResultConfig` controlling produced fields.
            seed: Optional random seed for this run.

        Returns:
            A completed `Job`. The job result includes metadata with the shot
            count, backend name, and effective result config. Validation
            failures raise directly; execution failures are captured in an
            error job whose `result()` re-raises.

        Raises:
            BackendValidationError: If requested fields are incompatible with
                the program or shot count.
            UnsupportedOperationError: If the program uses unsupported
                operations.

        Examples:
            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)

            result = qs.StateVectorBackend().run(
                program,
                shots=100,
                result_config=qs.ResultConfig(counts=True),
            ).result()
            counts = result.get_counts()
            ```
        """
        config = result_config if result_config is not None else ResultConfig()
        layout = self.resolve_layout(program)
        plan, facts = self._lower(program, layout)
        self._validate(config, shots, facts)
        try:
            return Job.done(
                self._execute(
                    config, shots, plan, facts, layout.n_qubits, layout.n_clbits, seed
                )
            )
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(
        self,
        config: ResultConfig,
        shots: int,
        facts: _PlanFacts,
    ) -> None:
        """Validate result-config / shots constraints against the lowered program.

        Operation support and dynamic classification were already resolved in
        `_lower`. Mid-circuit measurement and conditional operations are now
        supported, so they are not rejected here.
        """
        request = _resolve_result_request(config, facts)
        stochastic = facts.has_measurement or facts.has_reset
        requested_sv = config.statevector is True

        if (request.counts or (requested_sv and stochastic)) and type(shots) is not int:
            raise BackendValidationError(
                f"shots must be an int when requested results depend on it, got {shots!r}"
            )
        if request.counts and shots <= 0:
            raise BackendValidationError(f"counts require shots > 0, got shots={shots}")
        if requested_sv and stochastic and shots != 1:
            raise BackendValidationError(
                "statevector with measurement or reset is only supported for shots == 1"
            )

    # --- execution ---
    def _execute(
        self,
        config: ResultConfig,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        n_qubits: int,
        n_clbits: int,
        seed: int | None,
    ) -> Result:
        """Execute a lowered program and assemble the requested result fields."""
        request = _resolve_result_request(config, facts)

        if facts.is_dynamic:
            counts, statevector, available = self._run_per_shot(
                plan, n_qubits, n_clbits, shots, seed, request
            )
        else:
            counts, statevector, available = self._run_fast(
                plan,
                facts,
                n_qubits,
                n_clbits,
                shots,
                np.random.default_rng(seed),
                request,
            )

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if request.counts and "statevector" not in available:
            written = {
                c
                for s in plan
                if isinstance(s, MeasurementStep)
                for c in s.classical_indices
            }
            if any(c not in written for c in range(n_clbits)):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    NoMeasurementWarning,
                    stacklevel=3,
                )

        return Result(
            counts=counts,
            statevector=statevector,
            available=frozenset(available),
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "result_config": config,
            },
        )

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        n_qubits: int,
        n_clbits: int,
        shots: int,
        rng: np.random.Generator,
        request: _ResultRequest,
    ) -> tuple[dict[str, int] | None, np.ndarray | None, set[str]]:
        """Phase 1 path: evolve once, then sample terminal measurements."""
        engine = self._engine
        engine.initialize(n_qubits)
        measurements: list[tuple[int, int]] = []
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                engine.apply(step)
            else:  # MeasurementStep (no ResetStep on the fast path)
                measurements.extend(zip(step.measured_indices, step.classical_indices))

        counts: dict[str, int] | None = None
        statevector: np.ndarray | None = None
        available: set[str] = set()

        collapsed_index = None
        if request.statevector and facts.has_measurement:
            measured_qubits = [q for q, _c in measurements]
            collapsed_index = engine.collapse(measured_qubits, rng)

        if request.counts:
            if facts.has_measurement:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = engine.sample_indices(shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            counts = build_counts(indices, n_clbits, measurements)
            available.add("counts")

        if request.statevector:
            statevector = engine.export_state()
            available.add("statevector")

        return counts, statevector, available

    def _run_per_shot(
        self,
        plan: list[ResolvedStep],
        n_qubits: int,
        n_clbits: int,
        shots: int,
        seed: int | None,
        request: _ResultRequest,
    ) -> tuple[dict[str, int] | None, np.ndarray | None, set[str]]:
        """Per-shot path: run each shot independently with its own clbits.

        Each shot draws from its own child `SeedSequence`, spawned from the
        run's root seed up front and in shot order. Serial execution here
        consumes the same per-shot streams that parallel execution will use,
        so results stay identical regardless of how shots are distributed.
        """
        engine = self._engine
        # Counts need one trajectory per shot; statevector-only runs need one
        # representative trajectory, so shots=0 cannot leave the engine
        # uninitialized.
        n_iters = shots if request.counts else (1 if request.statevector else 0)
        seed_sequences = _shot_seed_sequences(seed, n_iters)

        ran_parallel = False
        if _should_parallelize(self._config, request, n_iters):
            snapshots = _run_dynamic_shots_parallel(
                self._config,
                plan,
                n_qubits,
                n_clbits,
                seed_sequences,
            )
            ran_parallel = True
        else:
            snapshots = []
            for seed_sequence in seed_sequences:
                rng = np.random.default_rng(seed_sequence)
                engine.initialize(n_qubits)
                snapshots.append(
                    _execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng)
                )

        counts: dict[str, int] | None = None
        statevector: np.ndarray | None = None
        available: set[str] = set()

        if request.counts:
            counts = build_counts_from_clbits(snapshots, n_clbits)
            available.add("counts")
        if request.statevector:
            if ran_parallel:
                raise BackendValidationError(
                    "statevector output is not available from parallel dynamic execution"
                )
            statevector = engine.export_state()
            available.add("statevector")

        return counts, statevector, available

    def _lower(
        self, program: Program, layout: ResourceLayout
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises `UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`. The pass also
        computes `is_dynamic` (reset, a condition, or a gate on an
        already-measured qubit), `has_measurement`, and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        measured_qubits: set[int] = set()
        is_dynamic = False
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                measured_indices = tuple(layout.qubit_index(q) for q in step.qreg)
                classical_indices = tuple(layout.clbit_index(c) for c in step.clreg)
                measured_qubits.update(measured_indices)
                plan.append(
                    MeasurementStep(
                        measured_indices=measured_indices,
                        classical_indices=classical_indices,
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                target_indices = tuple(layout.qubit_index(t) for t in step.targets)
                if step.condition is not None:
                    is_dynamic = True
                if any(t in measured_qubits for t in target_indices):
                    is_dynamic = True

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    is_dynamic = True
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(
                        ResetStep(reset_indices=target_indices, condition=cond)
                    )
                    continue

                rule = self._impl_map.get(type(step.operation))
                if rule is None:
                    raise UnsupportedOperationError(type(step.operation).__name__)
                matrix = rule(step)
                cond = _resolve_condition(step.condition, layout)
                plan.append(
                    ApplyMatrixStep(
                        matrix=matrix, target_indices=target_indices, condition=cond
                    )
                )

        return (
            plan,
            _PlanFacts(
                is_dynamic=is_dynamic,
                has_measurement=has_measurement,
                has_reset=has_reset,
            ),
        )
