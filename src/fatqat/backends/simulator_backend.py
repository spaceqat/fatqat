"""Unified matrix-family simulator backend with Qiskit-style method selection.

`SimulatorBackend` is the single entry point for matrix-family simulation:
``SimulatorBackend(method="statevector")`` and
``SimulatorBackend(method="density_matrix")`` (aliases ``"SV"`` / ``"DM"``,
case-insensitive) select the state representation, exactly like Qiskit Aer's
``AerSimulator(method=...)``. It is the only simulator backend:
per-representation backend classes do not exist.

Everything method-independent lives here once: options/result-config
normalization, lowering (including the compiler-facing `Barrier` skip),
validation, execution orchestration, and public `Result` assembly. The
method-dependent facts are bound as instance attributes in ``__init__`` -
the backend never branches on method afterwards:

- ``_state_field``: the native state field name (``"statevector"`` or
  ``"density_matrix"``). Drives the result-config flag read, the `Result`
  keyword, the availability name, the metadata echo, and validation wording.
- ``_simulator_cls``: the `Simulator` subclass this method drives
  (`NumpySVSimulator` or `NumpyDMSimulator`); one instance is bound to
  ``_simulator`` and reused across runs.
- ``_result_config_cls`` / ``_request_cls``: the method's frozen
  result-config and engine-request value objects. Supported
  ``result_config`` keys are derived from the config dataclass fields.
- ``_reset_is_stochastic``: whether reset makes execution stochastic for the
  state representation (`True` for statevector, where reset samples a
  branch; `False` for density matrix, where reset is a deterministic
  channel).

The backend/simulator seam: this class constructs the method's `Simulator`
subclass once, then calls ``simulator.initialize(system_dims, n_clbits)``
and ``simulator.run(plan, shots, seed, request) -> RawResult`` per run,
exactly as the per-method backends did against the engine before this
refactor.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from math import prod
from typing import Any

from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    UnsupportedOperationError,
)
from ..implementation import (
    MatrixImplementation,
    DeviceOperands,
    ImplementationMap,
    default_matrix_implementation_map,
)
from ..job import Job
from ..layout import ResourceLayout
from ..operations import BarrierGate, Measurement, Operation, ResetGate
from ..program import AppliedOperation, Program
from ..result import (
    Result,
    _DensityMatrixResultConfig,
    _StateVectorResultConfig,
    counts_dict_from_arrays,
)
from ..simulator import NumpyDMSimulator, NumpySVSimulator, Simulator
from .backend_utils import (
    _PlanFacts,
    _normalize_dict_options,
    _resolve_condition,
)
from .engine_contract import (
    _DensityMatrixResultRequest,
    _EngineConfig,
    _StateVectorResultRequest,
)
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep, ResolvedStep

# Canonical method names plus Qiskit-style short aliases, all case-insensitive.
_METHOD_ALIASES = {
    "statevector": "statevector",
    "sv": "statevector",
    "density_matrix": "density_matrix",
    "dm": "density_matrix",
}


class SimulatorBackend:
    """Matrix-family simulator backend for ``fatqat.Program`` execution.

    The simulation method selects the state representation and its
    semantics; everything else (supported operations, grouped measurement,
    feedforward conditions, reset, execution strategies, result handling) is
    method-independent:

    - ``method="statevector"`` (alias ``"SV"``): pure-state simulation. The
      native result field is ``statevector``. Reset samples a branch, so any
      reset makes execution stochastic and forces per-shot replay.
    - ``method="density_matrix"`` (alias ``"DM"``): exact mixed-state
      simulation. The native result field is ``density_matrix``. Reset is
      the deterministic partial-trace channel, so reset alone neither makes
      a program stochastic nor forces per-shot execution.

    Each run is classified into a fast path (evolve once, sample requested
    counts from the terminal measurement distribution) or a dynamic path
    (per-shot replay with an explicit classical register) when the program
    contains classical conditions, reuse of measured subsystems, or - under
    statevector semantics - reset.

    Backend constructor options affect only dynamic counts execution:

    - ``max_workers``: maximum worker processes for dynamic counts
      parallelism. ``None`` means automatic selection.
    - ``parallel_mode``: one of ``"auto"``, ``"serial"``, ``"multiprocessing"``,
      or ``"loky"``. ``"auto"`` prefers ``loky`` when available and otherwise
      uses ``multiprocessing``. ``"serial"`` disables process-based parallel
      execution.

    A backend instance reuses one simulator across runs, so it is efficient
    for repeated single-threaded use but is not safe for concurrent ``run()``
    calls.

    Examples:
        Density-matrix simulation, Qiskit style:

        >>> import fatqat as fq
        >>> program = fq.Program(1)
        >>> program.add(fq.ops.H, 0)
        >>> result = fq.backends.SimulatorBackend(method="DM").run(
        ...     program,
        ...     result_config={"counts": False, "density_matrix": True},
        ... ).result()
        >>> result.get_density_matrix()
        array([[0.5+0.j, 0.5+0.j],
               [0.5+0.j, 0.5+0.j]])
    """

    def __init__(
        self,
        method: str = "statevector",
        options: dict[str, Any] | None = None,
        implementation_map: ImplementationMap | None = None,
    ) -> None:
        """Create a simulator backend for the given method.

        Args:
            method: Simulation method: ``"statevector"`` or
                ``"density_matrix"``, or the case-insensitive short aliases
                ``"SV"`` / ``"DM"``.
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
        normalized = _METHOD_ALIASES.get(str(method).lower())
        if normalized is None:
            raise BackendValidationError(
                f"unsupported method={method!r}; expected one of "
                "'statevector'/'SV' or 'density_matrix'/'DM'"
            )
        # Method-dependent facts, bound once. This block is the single
        # dispatch point: the methods below read the bound attributes and
        # never branch on the method themselves.
        self._state_field = normalized
        self._simulator_cls: type[Simulator]
        if normalized == "statevector":
            self._result_config_cls = _StateVectorResultConfig
            self._request_cls = _StateVectorResultRequest
            self._simulator_cls = NumpySVSimulator
            # A pure state cannot represent the mixed post-reset ensemble, so
            # reset must sample one branch - a random event, like measurement.
            self._reset_is_stochastic = True
        else:
            self._result_config_cls = _DensityMatrixResultConfig
            self._request_cls = _DensityMatrixResultRequest
            self._simulator_cls = NumpyDMSimulator
            # A density matrix holds the full ensemble, so reset is the
            # deterministic channel |0><0| (x) Tr_target(rho): only
            # measurement (whose outcome is recorded) is stochastic.
            self._reset_is_stochastic = False

        config = _normalize_dict_options(
            options,
            {"max_workers", "parallel_mode"},
            _EngineConfig,
            "options",
            "backend",
            backend_name=type(self).__name__,
        )
        if implementation_map is None:
            implementation_map = default_matrix_implementation_map()
        self._impl_map = implementation_map.copy()
        # The simulator is constructed once and re-initialized per run so its
        # buffers can be reused. Because it holds per-run state, a single
        # backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._simulator = self._simulator_cls(config=config)
        self._simulator_system: tuple[tuple[int, ...], int] | None = None

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

        Resolves the program to the backend's flat layout, chooses an
        execution strategy, runs the circuit, and returns an eager ``Job``
        whose ``result()`` yields a ``Result``.

        ``result_config`` accepts ``counts`` plus the method's native state
        field (``statevector`` or ``density_matrix``), each tri-state:
        ``None`` (backend default), ``True`` (request), ``False`` (suppress).
        Counts default to measurement presence; the state field defaults to
        non-stochastic execution. Requesting the state field for a stochastic
        program requires ``shots == 1``. ``Result.metadata`` always includes
        ``shots``, ``backend_name``, ``method``, and the effective
        ``result_config``.

        Args:
            program: Program to execute.
            shots: Number of logical shots to run when counts are requested.
            result_config: Optional plain dictionary describing which result
                fields to produce; unknown keys are ignored with a warning.
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
                without a backend implementation, or one whose target key is
                illegal for this backend.
        """
        known_keys = {field.name for field in fields(self._result_config_cls)}
        config = _normalize_dict_options(
            result_config,
            known_keys,
            self._result_config_cls,
            "result_config",
            "result_config",
            backend_name=type(self).__name__,
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
    def _validate(self, config: Any, shots: int, facts: _PlanFacts) -> None:
        """Validate result-config / shots constraints against the lowered program.

        Operation support and dynamic classification were already resolved in
        `_lower`. Stochasticity is representation-dependent: measurement is
        always stochastic; reset only when ``_reset_is_stochastic``.
        """
        request = _resolve_result_request(
            config,
            facts,
            self._request_cls,
            self._state_field,
            self._reset_is_stochastic,
        )
        stochastic = facts.has_measurement or (
            self._reset_is_stochastic and facts.has_reset
        )
        requested_state = getattr(config, self._state_field) is True

        # shots is only checked when the result actually depends on it: counts
        # always sample per shot, and a stochastic state export needs shots==1
        # below. A non-stochastic state-only request ignores shots entirely
        # (see the engine's per-shot path), so any value - including 0 - is fine.
        if (request.counts or (requested_state and stochastic)) and type(
            shots
        ) is not int:
            raise BackendValidationError(
                f"shots must be an int when requested results depend on it, got {shots!r}"
            )
        if request.counts and shots <= 0:
            raise BackendValidationError(f"counts require shots > 0, got shots={shots}")
        if requested_state and stochastic and shots != 1:
            stochastic_sources = (
                "measurement or reset" if self._reset_is_stochastic else "measurement"
            )
            raise BackendValidationError(
                f"{self._state_field} with {stochastic_sources} is only supported "
                "for shots == 1"
            )

    # --- execution ---
    def _execute(
        self,
        config: Any,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        system_dims: tuple[int, ...],
        classical_dims: tuple[int, ...],
        n_clbits: int,
        seed: int | None,
    ) -> Result:
        """Execute a lowered program and assemble the requested result fields."""
        request = _resolve_result_request(
            config,
            facts,
            self._request_cls,
            self._state_field,
            self._reset_is_stochastic,
        )

        system_key = (tuple(system_dims), n_clbits)
        if self._simulator_system != system_key:
            self._simulator.initialize(system_dims, n_clbits)
            self._simulator_system = system_key

        raw = self._simulator.run(plan, shots, seed, request)
        counts = None
        state = raw.state
        state_requested = getattr(request, self._state_field)
        available: set[str] = set()
        if request.counts:
            counts = counts_dict_from_arrays(raw.outcome_keys, raw.outcome_counts)
            available.add("counts")
        if state_requested:
            available.add(self._state_field)

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if request.counts and self._state_field not in available:
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
            available=frozenset(available),
            classical_dims=classical_dims,
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "method": self._state_field,
                "result_config": {
                    "counts": config.counts,
                    self._state_field: getattr(config, self._state_field),
                },
            },
            **{self._state_field: state},
        )

    def _implementation_for(
        self, operation: Operation, device_operands: DeviceOperands
    ) -> MatrixImplementation:
        """Resolve the matrix rule for an operation on a device target key.

        Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` if the operation has no rule at
        all, or if it has rules but none for this target key — the message
        distinguishes the two.
        """
        if not self._impl_map.supports(operation):
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported by this backend"
            )
        rule = self._impl_map.implementation_for(
            operation, device_operands=device_operands
        )
        if rule is None:
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported on device operands {device_operands}"
            )
        return rule

    def _lower(
        self, program: Program, layout: ResourceLayout
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`; `Barrier`
        is recognized by type and skipped entirely - it is a compiler-facing
        marker with no simulation semantics, so it emits no step and cannot
        affect execution strategy or result defaults. The pass also computes
        `has_measurement` and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                measured_indices = tuple(
                    layout.subsystem_index(q) for q in step.targets
                )
                classical_indices = tuple(layout.clbit_index(c) for c in step.outputs)
                plan.append(
                    MeasurementStep(
                        measured_indices=measured_indices,
                        classical_indices=classical_indices,
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                if isinstance(step.operation, BarrierGate):
                    continue

                target_indices = tuple(layout.subsystem_index(t) for t in step.targets)

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(ResetStep(reset_indices=target_indices, condition=cond))
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


def _resolve_result_request(
    config: Any,
    facts: _PlanFacts,
    request_cls: type,
    state_field: str,
    reset_is_stochastic: bool,
) -> Any:
    """Resolve default result fields from config and lowered program facts.

    Counts default to measurement presence. The state field defaults to
    non-stochastic execution, where stochasticity is representation-dependent:
    measurement always; reset only when ``reset_is_stochastic`` (statevector
    reset samples a branch; density-matrix reset is a deterministic channel).
    """
    stochastic = facts.has_measurement or (reset_is_stochastic and facts.has_reset)
    counts = config.counts if config.counts is not None else facts.has_measurement
    state = getattr(config, state_field)
    if state is None:
        state = not stochastic
    return request_cls(counts=counts, **{state_field: state})
