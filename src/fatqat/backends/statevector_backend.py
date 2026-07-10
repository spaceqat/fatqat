"""Statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import warnings
from math import prod

from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    UnsupportedOperationError,
)
from ..implementation import (
    MatrixImplementation,
    MatrixImplementationMap,
    TargetKey,
    default_matrix_implementation_map,
)
from ..job import Job
from ..layout import ResourceLayout
from ..operations import Measurement, Operation, ResetGate
from ..program import AppliedOperation, Program
from ..result import (
    Result,
    _ResultConfig,
    counts_dict_from_arrays,
)
from .backend_utils import (
    _PlanFacts,
    _normalize_dict_options,
    _resolve_condition,
    _resolve_result_request,
)
from .engine_contract import _EngineConfig
from .statevector_engine import StateVectorEngine
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep, ResolvedStep


class StateVectorBackend:
    """Statevector backend for ``fatqat.Program`` execution.

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

    - ``max_workers``: maximum worker processes for dynamic counts
      parallelism. ``None`` means automatic selection.
    - ``parallel_mode``: one of ``"auto"``, ``"serial"``, ``"multiprocessing"``,
      or ``"loky"``. ``"auto"`` prefers ``loky`` when available and otherwise
      uses ``multiprocessing``. ``"serial"`` disables process-based parallel
      execution.

    A backend instance reuses one engine across runs, so it is efficient for
    repeated single-threaded use but is not safe for concurrent ``run()``
    calls.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        implementation_map: MatrixImplementationMap | None = None,
    ) -> None:
        """Create a statevector backend.

        Args:
            options: Optional execution-strategy options. Supported keys are
                ``max_workers`` and ``parallel_mode``; unknown keys are
                ignored with a warning. These options only affect the dynamic
                counts path and do not change numerical semantics.
            implementation_map: Optional matrix implementation map controlling
                which operations this backend supports and how their matrices
                are built. ``None`` (the default) uses
                ``default_matrix_implementation_map()``. The backend copies
                whatever map it receives, so mutating the caller's map object
                after construction does not change this backend's behavior.
        """
        config = _normalize_dict_options(
            options, {"max_workers", "parallel_mode"}, _EngineConfig, "options", "backend"
        )
        if implementation_map is None:
            implementation_map = default_matrix_implementation_map()
        self._impl_map = implementation_map.copy()
        # The engine is constructed once and re-initialized per run so its
        # compiled kernels can be reused. Because it holds per-run state, a
        # single backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._engine = StateVectorEngine(config)
        self._engine_system: tuple[tuple[int, ...], int] | None = None

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
        runs the circuit, and returns an eager ``Job`` whose ``result()``
        yields a ``Result``.

        Result selection via ``result_config``:

        - ``{"counts": None}``: counts are produced when the program contains
          at least one measurement.
        - ``{"counts": True}``: counts are always requested.
        - ``{"counts": False}``: counts are suppressed, even if the program
          measures subsystems.
        - ``{"statevector": None}``: a statevector is produced only when
          execution is non-stochastic, meaning the program contains no
          measurement and no reset.
        - ``{"statevector": True}``: explicitly request a final statevector.
        - ``{"statevector": False}``: suppress statevector output.

        Output consequences:

        - Counts are returned through ``Result.get_counts()`` as
          little-endian classical count-key strings.
        - A statevector, when produced, is returned through
          ``Result.get_statevector()``.
        - If a field was not produced, its accessor raises
          ``ResultFieldUnavailableError``.
        - ``Result.metadata`` always includes ``shots``, ``backend_name``,
          and the effective ``result_config``.

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
          requested, multiple iterations are needed, and backend options
          allow it, shots may be distributed across worker processes. The
          counts are reproducible for a fixed ``seed`` regardless of serial
          vs parallel scheduling.

        Statevector semantics:

        - For non-stochastic programs, a produced statevector is the final
          evolved state after all operations.
        - For stochastic programs (any measurement or reset),
          ``statevector=True`` is only supported for ``shots == 1``; the
          returned statevector is the single-shot post-measurement/post-reset
          state.
        - A program may take the dynamic execution path yet still be
          non-stochastic, for example when it contains only classical
          conditions on never-written clbits. Such a program may still
          produce a default statevector.

        Shot semantics and validation:

        - ``shots`` matters whenever counts are requested.
        - ``shots`` must be an ``int`` whenever requested results depend on
          it.
        - Counts require ``shots > 0``.
        - Requesting a statevector for a stochastic program requires
          ``shots == 1``.

        Args:
            program: Program to execute.
            shots: Number of logical shots to run when counts are requested.
                For statevector-only deterministic execution, the value may be
                ignored.
            result_config: Optional plain dictionary describing which result
                fields to produce. Supported keys are ``counts`` and
                ``statevector``; unknown keys are ignored with a warning.
                When omitted, backend defaults are used.
            seed: Optional root seed for the run. For dynamic counts, one
                reproducible child RNG stream is derived per logical shot.

        Returns:
            A completed ``Job``. Validation failures raise directly from
            ``run()``; execution-stage failures are captured in a failed job
            whose ``result()`` re-raises the underlying exception.

        Raises:
            BackendValidationError: If requested outputs are incompatible with
                the program shape or ``shots``.
            UnsupportedOperationError: If the program contains an operation
                without a backend implementation, or one whose target key
                (e.g. a non-neighbor qubit pair) is illegal for this backend.

        Examples:
            Sample counts from a measured program:

            .. code-block:: python

                import fatqat as fq

                program = fq.Program(1, 1)
                program.add(fq.ops.X, 0)
                program.add_measurement(0, 0)

                result = fq.backends.StateVectorBackend().run(
                    program,
                    shots=100,
                    result_config={"counts": True},
                ).result()
                counts = result.get_counts()

            Request a deterministic statevector:

            .. code-block:: python

                program = fq.Program(1)
                program.add(fq.ops.H, 0)

                result = fq.backends.StateVectorBackend().run(
                    program,
                    result_config={"counts": False, "statevector": True},
                ).result()
                statevector = result.get_statevector()
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

        system_key = (tuple(system_dims), n_clbits)
        if self._engine_system != system_key:
            self._engine.initialize(system_dims, n_clbits)
            self._engine_system = system_key

        raw = self._engine.run(plan, shots, seed, request)
        counts = None
        statevector = raw.state
        available: set[str] = set()
        if request.counts:
            counts = counts_dict_from_arrays(raw.outcome_keys, raw.outcome_counts)
            available.add("counts")
        if request.statevector:
            available.add("statevector")

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

    def _implementation_for(
        self, operation: Operation, target_key: TargetKey
    ) -> MatrixImplementation:
        """Resolve the matrix rule for an operation on a device target key.

        Both failure cases a target-aware map can report — an operation
        family with no rule at all, or a supported family on an illegal
        target key — raise the same `UnsupportedOperationError`, with a
        message specific to which one occurred; callers that only need to
        know "this can't run" don't need to know the two cases exist, and
        `UnsupportedOperationError` is a `BackendValidationError` for
        callers that catch the broader family. For a `register`-only map
        (no target-aware data), `get(operation, target_key)` returns the
        class-keyed rule for every target key, so this behaves exactly like
        a bare `get(operation)` lookup.
        """
        if not self._impl_map.supports(operation):
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported by this backend"
            )
        rule = self._impl_map.get(operation, target_key)
        if rule is None:
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported on target key {target_key}"
            )
        return rule

    def _lower(
        self, program: Program, layout: ResourceLayout
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises `UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`. The pass also
        computes `has_measurement` and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                measured_indices = tuple(layout.subsystem_index(q) for q in step.qreg)
                classical_indices = tuple(layout.clbit_index(c) for c in step.clreg)
                plan.append(
                    MeasurementStep(
                        measured_indices=measured_indices,
                        classical_indices=classical_indices,
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                target_indices = tuple(layout.subsystem_index(t) for t in step.targets)

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(
                        ResetStep(reset_indices=target_indices, condition=cond)
                    )
                    continue

                rule = self._implementation_for(step.operation, target_indices)
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
                has_measurement=has_measurement,
                has_reset=has_reset,
            ),
        )
