"""Statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import importlib.util
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from math import prod
from typing import Any

import numpy as np

from .engine import StateVectorEngine
from .errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    UnsupportedOperationError,
)
from .implementation import ApplyMatrixStep, MatrixImplementationMap, default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .operations import ResetGate
from .program import AppliedOperation, Measurement, Program
from .result import (
    Result,
    _ResultConfig,
    build_counts,
    build_counts_from_clbits,
)


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat subsystem indices into matching flat clbit indices."""

    measured_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]


@dataclass(frozen=True)
class ResetStep:
    """Resolved reset of one or more flat subsystems to |0>, with optional condition.

    `Reset` is an `AppliedOperation`, so it can carry a feedforward `condition`
    just like a gate. The lowered form stores it as ``(clbit_index, value)``
    AND-terms; the per-shot loop skips the reset when the guard fails.
    """

    reset_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None


ResolvedStep = ApplyMatrixStep | MeasurementStep | ResetStep


@dataclass(frozen=True)
class _PlanFacts:
    """Classification facts computed while lowering a program.

    `is_dynamic` picks the execution strategy (per-shot loop vs. one-shot
    evolve-then-sample); it is true for a condition, reset, or measurement.
    This is a different axis from `stochastic` (see `_resolve_result_request`)
    - a condition-only program is dynamic but not stochastic.
    """

    is_dynamic: bool
    has_measurement: bool
    has_reset: bool


@dataclass(frozen=True)
class _ResultRequest:
    """Resolved result fields requested for one execution."""

    counts: bool
    statevector: bool


_PARALLEL_MODE_NAMES = frozenset({"auto", "serial", "multiprocessing", "loky"})


@dataclass(frozen=True)
class _BackendConfig:
    """Normalized backend execution-strategy options.

    Option *values* are validated here, at construction, so an invalid
    `max_workers` or `parallel_mode` fails from `StateVectorBackend(...)`
    rather than being deferred to a run and swallowed into a failed `Job`.
    """

    max_workers: Any = None
    parallel_mode: Any = "auto"

    def __post_init__(self) -> None:
        mw = self.max_workers
        if mw is not None and (type(mw) is not int or mw < 1):
            raise BackendValidationError(
                f"max_workers must be a positive int or None, got {mw!r}"
            )
        if self.parallel_mode not in _PARALLEL_MODE_NAMES:
            raise BackendValidationError(
                f"unsupported parallel_mode={self.parallel_mode!r}"
            )


def _normalize_dict_options(
    options: dict[str, Any] | None,
    known_keys: set[str],
    config_cls: type,
    param_name: str,
    warning_noun: str,
) -> Any:
    """Normalize a plain dict of options into a frozen config dataclass.

    `None` returns `config_cls()` (all defaults). A non-dict, non-`None` value
    raises `TypeError`. Unknown keys are dropped with an aggregated warning;
    known keys are passed through to `config_cls`, so any key the caller
    omits falls back to that dataclass field's own default.

    Shared by `StateVectorBackend.__init__`'s `options` and `run`'s
    `result_config` so both dict-configured surfaces validate and warn
    identically instead of drifting (see `_BackendConfig`/`_ResultConfig`).
    """
    if options is None:
        return config_cls()
    if not isinstance(options, dict):
        raise TypeError(f"{param_name} must be a dict or None, got {type(options)!r}")
    known = {key: value for key, value in options.items() if key in known_keys}
    ignored = {key: value for key, value in options.items() if key not in known_keys}
    if ignored:
        warnings.warn(
            f"StateVectorBackend ignored unsupported {warning_noun} options: {ignored!r}",
            UserWarning,
            stacklevel=3,
        )
    return config_cls(**known)


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    layout: ResourceLayout,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    # `val` is already an int (Program._normalize_condition guarantees it); no
    # re-coercion here.
    return tuple((layout.clbit_index(ref), val) for ref, val in condition)


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
            bits = engine.measure_subsystems(step.measured_indices, rng)
            for c, bit in zip(step.classical_indices, bits):
                clbits[c] = bit
        else:  # ResetStep
            if _condition_matches(step.condition, clbits):
                engine.reset_subsystems(step.reset_indices, rng)
    return tuple(clbits)


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


def _planned_workers(
    config: _BackendConfig,
    request: _ResultRequest,
    n_iters: int,
) -> int | None:
    if not request.counts:
        return None
    if n_iters <= 1:
        return None
    if config.parallel_mode == "serial":
        return None
    if config.max_workers is None and n_iters < _PARALLEL_MIN_SHOTS:
        # Automatic mode stays serial for small runs; explicit max_workers wins.
        return None
    workers = min(_effective_max_workers(config.max_workers, n_iters), n_iters)
    return workers if workers > 1 else None


def _resolve_parallel_mode_name(name: str) -> str:
    """Resolve `"auto"` to a concrete backend; other names are passed through.

    `name` is already one of the validated `_PARALLEL_MODE_NAMES` (checked
    in `_BackendConfig.__post_init__`), so this only expands `"auto"`.
    """
    if name == "auto":
        return "loky" if _loky_available() else "multiprocessing"
    return name


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
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_batch: list[np.random.SeedSequence],
) -> list[tuple[int, ...]]:
    """Run a contiguous batch of dynamic shots in one worker with one engine."""
    engine = StateVectorEngine()
    snapshots: list[tuple[int, ...]] = []
    for seed_sequence in seed_batch:
        rng = np.random.default_rng(seed_sequence)
        engine.initialize(system_dims)
        snapshots.append(_execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng))
    return snapshots


def _run_dynamic_shots_multiprocessing(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    batches = _split_into_batches(seed_sequences, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _run_dynamic_shot_batch,
            repeat(plan),
            repeat(system_dims),
            repeat(n_clbits),
            batches,
        )
        return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_loky(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
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
        repeat(system_dims),
        repeat(n_clbits),
        batches,
    )
    return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_parallel(
    config: _BackendConfig,
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    """Dispatch shots to worker processes. Caller must supply `max_workers`.

    `max_workers` is the same value `_planned_workers` already returned to
    decide whether to call this function at all; it is passed through rather
    than recomputed here, so there is exactly one source of truth for the
    parallel/serial decision.
    """
    mode_name = _resolve_parallel_mode_name(config.parallel_mode)
    if mode_name == "loky":
        if _loky_available():
            return _run_dynamic_shots_loky(
                plan, system_dims, n_clbits, seed_sequences, max_workers
            )
        warnings.warn(
            "parallel_mode='loky' requested but loky is unavailable; "
            "falling back to multiprocessing",
            UserWarning,
            stacklevel=3,
        )
    # "multiprocessing", plus the loky-unavailable fallback above.
    return _run_dynamic_shots_multiprocessing(
        plan, system_dims, n_clbits, seed_sequences, max_workers
    )


def _resolve_result_request(config: _ResultConfig, facts: _PlanFacts) -> _ResultRequest:
    """Resolve default result fields from config and lowered program facts."""
    # stochastic: whether shots can actually differ from each other. Only
    # measurement/reset can cause that; a bare condition just gates whether an
    # operation applies, so a condition-only program is never stochastic.
    stochastic = facts.has_measurement or facts.has_reset
    counts = config.counts if config.counts is not None else facts.has_measurement
    statevector = config.statevector
    if statevector is None:
        statevector = not stochastic
    return _ResultRequest(counts=counts, statevector=statevector)


class StateVectorBackend:
    """Statevector backend for `qnsim.Program` execution.

    The backend supports matrix-evolvable gates, grouped measurement,
    feedforward conditions, and reset. Each run is classified into one of two
    execution strategies:

    - Fast path: used when the program has no reset, no classically
      conditioned operations, and no operation that acts on a subsystem after
      that subsystem has been measured. The statevector is evolved once;
      requested counts are then sampled from the resulting measurement
      distribution.
    - Dynamic path: used when the program contains reset, a classical
      condition, or reuse of a measured subsystem. The backend executes one
      shot at a time while tracking the classical register explicitly,
      because later operations may depend on earlier measurement outcomes.

    Backend constructor options affect only dynamic counts execution:

    - `max_workers`: maximum worker processes for dynamic counts parallelism.
      `None` means automatic selection.
    - `parallel_mode`: one of `"auto"`, `"serial"`, `"multiprocessing"`,
      or `"loky"`. `"auto"` prefers `loky` when available and otherwise uses
      `multiprocessing`. `"serial"` disables process-based parallel execution.

    A backend instance reuses one engine across runs, so it is efficient for
    repeated single-threaded use but is not safe for concurrent `run()` calls.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        implementation_map: MatrixImplementationMap | None = None,
    ) -> None:
        """Create a statevector backend.

        Args:
            options: Optional execution-strategy options. Supported keys are
                `max_workers` and `parallel_mode`; unknown keys are ignored
                with a warning. These options only affect the dynamic counts
                path and do not change numerical semantics.
            implementation_map: Optional matrix implementation map controlling
                which operations this backend supports and how their matrices
                are built. `None` (the default) uses
                `default_implementation_map()`. The backend copies whatever
                map it receives, so mutating the caller's map object after
                construction does not change this backend's behavior.
        """
        self._config = _normalize_dict_options(
            options, {"max_workers", "parallel_mode"}, _BackendConfig, "options", "backend"
        )
        if implementation_map is None:
            implementation_map = default_implementation_map()
        self._impl_map = implementation_map.copy()
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
        result_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Job:
        """Validate, execute, and package one program run.

        This is the main user-facing execution entry point. It resolves the
        program to the backend's flat layout, chooses an execution strategy,
        runs the circuit, and returns an eager `Job` whose `result()` yields a
        `Result`.

        Result selection via `result_config`:

        - `{"counts": None}`: counts are produced when the program contains at
          least one measurement.
        - `{"counts": True}`: counts are always requested.
        - `{"counts": False}`: counts are
          suppressed, even if the
          program measures subsystems.
        - `{"statevector": None}`: a statevector is produced only when
          execution is non-stochastic, meaning the program contains no
          measurement and no reset.
        - `{"statevector": True}`: explicitly request a final statevector.
        - `{"statevector": False}`:
          suppress statevector output.

        Output consequences:

        - Counts are returned through `Result.get_counts()` as little-endian
          classical count-key strings.
        - A statevector, when produced, is returned through
          `Result.get_statevector()`.
        - If a field was not produced, its accessor raises
          `ResultFieldUnavailableError`.
        - `Result.metadata` always includes `shots`, `backend_name`, and the
          effective `result_config`.

        Execution strategy:

        - Fast path: programs without reset, classical conditions, or reuse of
          a measured subsystem are evolved once. Requested counts are sampled
          from terminal measurement mappings without replaying the full
          circuit shot by shot.
        - Dynamic path: programs with reset, classical conditions, or reuse of
          measured subsystems are executed shot by shot with an explicit
          classical register. This path preserves feedforward semantics and
          repeated measurement/reset behavior.
        - Parallel dynamic counts: when the dynamic path is used, counts are
          requested, multiple iterations are needed, and backend options allow
          it, shots may be distributed across worker processes. The counts are
          reproducible for a fixed `seed` regardless of serial vs parallel
          scheduling.

        Statevector semantics:

        - For non-stochastic programs, a produced statevector is the final
          evolved state after all operations.
        - For stochastic programs (any measurement or reset),
          `statevector=True` is only supported for `shots == 1`; the returned
          statevector is the single-shot post-measurement/post-reset state.
        - A program may take the dynamic execution path yet still be
          non-stochastic, for example when it contains only classical
          conditions on never-written clbits. Such a program may still produce
          a default statevector.

        Shot semantics and validation:

        - `shots` matters whenever counts are requested.
        - `shots` must be an `int` whenever requested results depend on it.
        - Counts require `shots > 0`.
        - Requesting a statevector for a stochastic program requires
          `shots == 1`.

        Args:
            program: Program to execute.
            shots: Number of logical shots to run when counts are requested.
                For statevector-only deterministic execution, the value may be
                ignored.
            result_config: Optional plain dictionary describing which result
                fields to produce. Supported keys are `counts` and
                `statevector`; unknown keys are ignored with a warning. When
                omitted, backend defaults are used.
            seed: Optional root seed for the run. For dynamic counts, one
                reproducible child RNG stream is derived per logical shot.

        Returns:
            A completed `Job`. Validation failures raise directly from `run()`;
            execution-stage failures are captured in a failed job whose
            `result()` re-raises the underlying exception.

        Raises:
            BackendValidationError: If requested outputs are incompatible with
                the program shape or `shots`.
            UnsupportedOperationError: If the program contains an operation
                without a backend implementation.

        Examples:
            Sample counts from a measured program:

            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)

            result = qs.backends.StateVectorBackend().run(
                program,
                shots=100,
                result_config={"counts": True},
            ).result()
            counts = result.get_counts()
            ```

            Request a deterministic statevector:

            ```python
            program = qs.Program(1)
            program.add(qs.ops.H, 0)

            result = qs.backends.StateVectorBackend().run(
                program,
                result_config={"counts": False, "statevector": True},
            ).result()
            statevector = result.get_statevector()
            ```
        """
        config = _normalize_dict_options(
            result_config, {"counts", "statevector"}, _ResultConfig, "result_config", "result_config"
        )
        layout = self.resolve_layout(program)
        plan, facts = self._lower(program, layout)
        self._validate(config, shots, facts)
        try:
            return Job.done(
                self._execute(
                    config,
                    shots,
                    plan,
                    facts,
                    layout.system_dims,
                    layout.classical_dims,
                    layout.n_clbits,
                    seed,
                )
            )
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(
        self,
        config: _ResultConfig,
        shots: int,
        facts: _PlanFacts,
    ) -> None:
        """Validate result-config / shots constraints against the lowered program.

        Operation support and dynamic classification were already resolved in
        `_lower`. Mid-circuit measurement and conditional operations are now
        supported, so they are not rejected here.
        """
        request = _resolve_result_request(config, facts)
        stochastic = facts.has_measurement or facts.has_reset  # see _resolve_result_request
        requested_sv = config.statevector is True

        # shots is only checked when the result actually depends on it: counts
        # always sample per shot, and a stochastic statevector needs shots==1
        # below. A non-stochastic statevector-only request ignores shots
        # entirely (see _run_per_shot), so any value - including 0 - is fine.
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
        config: _ResultConfig,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        system_dims: tuple[int, ...],
        classical_dims: tuple[int, ...],
        n_clbits: int,
        seed: int | None,
    ) -> Result:
        """Execute a lowered program and assemble the requested result fields."""
        request = _resolve_result_request(config, facts)

        if facts.is_dynamic:
            counts, statevector, available = self._run_per_shot(
                plan, system_dims, n_clbits, shots, seed, request
            )
        else:
            counts, statevector, available = self._run_fast(
                plan,
                facts,
                system_dims,
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
            classical_dims=classical_dims,
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "result_config": {
                    "counts": config.counts,
                    "statevector": config.statevector,
                },
            },
        )

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        system_dims: tuple[int, ...],
        n_clbits: int,
        shots: int,
        rng: np.random.Generator,
        request: _ResultRequest,
    ) -> tuple[dict[tuple[int, ...], int] | None, np.ndarray | None, set[str]]:
        """Phase 1 path: evolve once, then sample terminal measurements."""
        engine = self._engine
        engine.initialize(system_dims)
        measurements: list[tuple[int, int]] = []
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                engine.apply(step)
            else:  # MeasurementStep (no ResetStep on the fast path)
                measurements.extend(zip(step.measured_indices, step.classical_indices))

        counts: dict[tuple[int, ...], int] | None = None
        statevector: np.ndarray | None = None
        available: set[str] = set()

        collapsed_index = None
        if request.statevector and facts.has_measurement:
            measured_subsystems = [q for q, _c in measurements]
            collapsed_index = engine.collapse(measured_subsystems, rng)

        if request.counts:
            if facts.has_measurement:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = engine.sample_indices(shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            counts = build_counts(indices, n_clbits, measurements, system_dims)
            available.add("counts")

        if request.statevector:
            statevector = engine.export_state()
            available.add("statevector")

        return counts, statevector, available

    def _run_per_shot(
        self,
        plan: list[ResolvedStep],
        system_dims: tuple[int, ...],
        n_clbits: int,
        shots: int,
        seed: int | None,
        request: _ResultRequest,
    ) -> tuple[dict[tuple[int, ...], int] | None, np.ndarray | None, set[str]]:
        """Per-shot path: run each shot independently with its own clbits.

        Each shot draws from its own child `SeedSequence`, spawned from the
        run's root seed up front and in shot order. Serial execution here
        consumes the same per-shot streams that parallel execution will use,
        so results stay identical regardless of how shots are distributed.

        A requested statevector forces the serial path: the exported state is
        read off this backend's single engine after the last trajectory, which
        worker processes do not share. Any program reaching here with a
        statevector request is non-stochastic (measurement/reset with a
        statevector is rejected for `shots > 1` in `_validate`), so every
        trajectory produces the same state and the serial run stays correct.
        """
        engine = self._engine
        # Counts need one trajectory per shot; statevector-only runs need one
        # representative trajectory, so shots=0 cannot leave the engine
        # uninitialized.
        n_iters = shots if request.counts else (1 if request.statevector else 0)
        seed_sequences = _shot_seed_sequences(seed, n_iters)

        max_workers = (
            None if request.statevector
            else _planned_workers(self._config, request, n_iters)
        )
        if max_workers is not None:
            snapshots = _run_dynamic_shots_parallel(
                self._config,
                plan,
                system_dims,
                n_clbits,
                seed_sequences,
                max_workers,
            )
        else:
            snapshots = []
            for seed_sequence in seed_sequences:
                rng = np.random.default_rng(seed_sequence)
                engine.initialize(system_dims)
                snapshots.append(
                    _execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng)
                )

        counts: dict[tuple[int, ...], int] | None = None
        statevector: np.ndarray | None = None
        available: set[str] = set()

        if request.counts:
            counts = build_counts_from_clbits(snapshots, n_clbits)
            available.add("counts")
        if request.statevector:
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
        already-measured subsystem), `has_measurement`, and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        measured_subsystems: set[int] = set()
        is_dynamic = False
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                measured_indices = tuple(layout.subsystem_index(q) for q in step.qreg)
                classical_indices = tuple(layout.clbit_index(c) for c in step.clreg)
                measured_subsystems.update(measured_indices)
                plan.append(
                    MeasurementStep(
                        measured_indices=measured_indices,
                        classical_indices=classical_indices,
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                target_indices = tuple(layout.subsystem_index(t) for t in step.targets)
                if step.condition is not None:
                    is_dynamic = True
                if any(t in measured_subsystems for t in target_indices):
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
                try:
                    matrix = rule(step.operation, targets=step.targets)
                except Exception as exc:
                    raise MatrixImplementationError(
                        f"implementation for {type(step.operation).__name__} raised: {exc}"
                    ) from exc

                # Check matrix shape matches target dimensions
                target_dims = tuple(layout.system_dims[i] for i in target_indices)
                expected = prod(target_dims)
                if matrix.shape != (expected, expected):
                    raise BackendValidationError(
                        f"{type(step.operation).__name__} resolved to a "
                        f"{matrix.shape} matrix, incompatible with target "
                        f"dimensions {target_dims} (expected "
                        f"{(expected, expected)})"
                    )

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
