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
- ``_simulator_cls``: the `Simulator` subclass the (method, runtime) pair
  drives (`NumpySVSimulator`, `NumpyDMSimulator`, or the optional
  `NumbaSVSimulator`); one instance is bound to ``_simulator`` and reused
  across runs.
- ``_result_config_cls`` / ``_request_cls``: the method's frozen
  result-config and engine-request value objects. Supported
  ``result_config`` keys are derived from the config dataclass fields.
- ``_nonunitary_is_stochastic``: whether non-unitary maps (reset, channel
  noise) make execution stochastic for the state representation (`True`
  for statevector, which must sample one branch of any non-unitary map;
  `False` for density matrix, which applies them as deterministic
  channels).

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
from ..flat_layout import FlatResourceLayout
from ..implementation import (
    MatrixImplementation,
    DeviceOperands,
    ImplementationMap,
    default_matrix_implementation_map,
)
from ..job import Job
from ..noise import (
    ChannelImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    default_channel_implementation_map,
)
from ..noise.base import _validate_kraus_shapes
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
from .resource_binding import ResourceBinding, _scalar_identity_binder
from .steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    MeasurementStep,
    ResetStep,
    ResolvedStep,
)

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
    statevector semantics - reset or channel noise.

    Channel noise: a :py:class:`~fatqat.NoiseModel` passed via ``noise=``
    attaches Kraus channels to gate occurrences; each is resolved at lowering
    and applied right after its gate. Under density-matrix semantics a
    channel is the exact Kraus sum (deterministic, fast-path compatible);
    under statevector semantics each occurrence samples one trajectory
    branch, which makes execution stochastic and forces per-shot replay.

    Backend constructor options affect only dynamic counts execution:

    - ``max_workers``: maximum worker processes for dynamic counts
      parallelism. ``None`` means automatic selection.
    - ``parallel_mode``: one of ``"auto"``, ``"serial"``, ``"multiprocessing"``,
      or ``"loky"``. ``"auto"`` prefers ``loky`` when available and otherwise
      uses ``multiprocessing``. ``"serial"`` disables process-based parallel
      execution.

    The ``runtime`` argument selects the execution technology for the chosen
    representation - ``"numpy"`` (default) or ``"numba"`` (optional
    dependency, statevector only for now). The runtime never changes
    simulation semantics, only how fast the same numbers are computed;
    dynamic-shot worker processes use the selected runtime as well.

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
        runtime: str = "numpy",
        noise: NoiseModel | None = None,
        channel_implementation_map: ChannelImplementationMap | None = None,
    ) -> None:
        """Create a simulator backend for the given method and runtime.

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
            runtime: Execution technology: ``"numpy"`` (the default) or
                ``"numba"``, case-insensitive. The runtime selects *how* the
                chosen state representation is computed, never its semantics:
                results are identical up to the documented per-simulator RNG
                reproducibility contract. ``"numba"`` currently supports
                ``method="statevector"`` only and requires the optional
                ``numba`` dependency; both constraints raise here, at
                construction, rather than at run time.
            noise: Optional :py:class:`~fatqat.NoiseModel` applied to every
                run. ``None`` (the default) means noise-free execution. The
                backend holds a reference (not a copy): a noise model is
                standalone, reusable state the user may keep building.
            channel_implementation_map: Optional map controlling which
                `Channel` descriptor types this backend can resolve and how
                their Kraus operators are built. ``None`` (the default) uses
                ``default_channel_implementation_map()``. Copied, like
                ``implementation_map``.

        Raises:
            BackendValidationError: If ``method`` or ``runtime`` is not one
                of the supported names, ``runtime="numba"`` is combined with
                ``method="density_matrix"`` (no numba density-matrix
                simulator exists yet), or numba is not installed.
        """
        normalized = _METHOD_ALIASES.get(str(method).lower())
        if normalized is None:
            raise BackendValidationError(
                f"unsupported method={method!r}; expected one of "
                "'statevector'/'SV' or 'density_matrix'/'DM'"
            )
        normalized_runtime = str(runtime).lower()
        if normalized_runtime not in ("numpy", "numba"):
            raise BackendValidationError(
                f"unsupported runtime={runtime!r}; expected 'numpy' or 'numba'"
            )
        # Method- and runtime-dependent facts, bound once. This block is the
        # single dispatch point: the methods below read the bound attributes
        # and never branch on method or runtime themselves.
        self._state_field = normalized
        self._simulator_cls: type[Simulator]
        if normalized == "statevector":
            self._result_config_cls = _StateVectorResultConfig
            self._request_cls = _StateVectorResultRequest
            self._simulator_cls = NumpySVSimulator
            # A pure state cannot represent the mixed output of a non-unitary
            # map, so reset and channel noise must each sample one branch - a
            # random event, like measurement.
            self._nonunitary_is_stochastic = True
        else:
            self._result_config_cls = _DensityMatrixResultConfig
            self._request_cls = _DensityMatrixResultRequest
            self._simulator_cls = NumpyDMSimulator
            # A density matrix holds the full ensemble, so reset is the
            # deterministic channel |0><0| (x) Tr_target(rho) and channel
            # noise is the exact Kraus sum: only measurement (whose outcome
            # is recorded) is stochastic.
            self._nonunitary_is_stochastic = False
        if normalized_runtime == "numba":
            # The runtime axis swaps the simulator class only; every other
            # method-bound fact above is representation semantics and stays.
            if normalized == "density_matrix":
                raise BackendValidationError(
                    "runtime='numba' does not support method='density_matrix' "
                    "yet; use runtime='numpy' for density-matrix simulation"
                )
            try:
                # Lazy: numba is an optional dependency, and fatqat.simulator's
                # package __init__ deliberately never imports the nb module.
                from ..simulator.nb import NumbaSVSimulator
            except ImportError as exc:
                raise BackendValidationError(
                    "runtime='numba' requires the optional numba dependency "
                    "(install the 'numba' group)"
                ) from exc
            self._simulator_cls = NumbaSVSimulator
        self._runtime = normalized_runtime

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
        # The noise model is held by reference (it is standalone, reusable
        # user state); the channel map is copied, like the matrix map.
        self._noise_model = noise if noise is not None else NoiseModel()
        if channel_implementation_map is None:
            channel_implementation_map = default_channel_implementation_map()
        self._channel_map = channel_implementation_map.copy()
        # The simulator is constructed once and re-initialized per run so its
        # buffers can be reused. Because it holds per-run state, a single
        # backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._simulator = self._simulator_cls(config=config)
        self._simulator_system: tuple[tuple[int, ...], int] | None = None

    def resolve_layout(self, program: Program) -> FlatResourceLayout:
        """Build the flat resource layout used by this backend.

        Args:
            program: Program whose registers should be flattened.

        Returns:
            Resource layout mapping register references to flat indices.
        """
        return FlatResourceLayout.from_program(program)

    def _create_resource_binding(
        self, program: Program, flat_layout: FlatResourceLayout
    ) -> ResourceBinding:
        """Build the resource binding used to resolve this run's targets.

        Protected extension hook, called once per run (after
        `resolve_layout`, before `_lower`) rather than stored as mutable
        per-run backend state. The default installs only the scalar/identity
        binder: a `RegisterRef` resolves to a `BoundResource` whose
        `engine_index` and `device_label` are both `flat_layout.subsystem_index(ref)`.
        A `RegisterView` is declined by this binder and, since the base
        backend installs no other, raises
        `~fatqat.errors.UnsupportedResourceOperandError` from `_lower`.

        Override to install additional binders (tried before the scalar
        binder) for richer frontend resource types.
        """
        return ResourceBinding([_scalar_identity_binder])

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
            UnsupportedResourceOperandError: If a unitary gate's target
                cannot be resolved by any resource binder this backend
                installs (e.g. a ``RegisterView`` on a backend that binds
                only scalar ``RegisterRef`` targets).
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
        binding = self._create_resource_binding(program, layout)
        plan, facts = self._lower(program, layout, binding)
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
        always stochastic; reset and channel noise only when
        ``_nonunitary_is_stochastic``.
        """
        request = _resolve_result_request(
            config,
            facts,
            self._request_cls,
            self._state_field,
            self._nonunitary_is_stochastic,
        )
        stochastic = facts.has_measurement or (
            self._nonunitary_is_stochastic and (facts.has_reset or facts.has_channel)
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
                "measurement, reset, or channel noise"
                if self._nonunitary_is_stochastic
                else "measurement"
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
            self._nonunitary_is_stochastic,
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
                "runtime": self._runtime,
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
        self,
        program: Program,
        layout: FlatResourceLayout,
        binding: ResourceBinding | None = None,
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`; `Barrier`
        is recognized by type and skipped entirely - it is a compiler-facing
        marker with no simulation semantics, so it emits no step and cannot
        affect execution strategy or result defaults. Channels the noise
        model attaches to a gate occurrence are resolved here into
        `ApplyChannelStep`s inserted right after the gate's own step, one per
        channel in registration order, each inheriting the gate's condition.
        The pass also computes `has_measurement`, `has_reset`, and
        `has_channel`.

        `Measurement`, `Reset`, and `Barrier` targets are always scalar
        `RegisterRef`s (views are frontend-rejected for them) and are
        resolved directly via `layout.subsystem_index`, unchanged. Only the
        unitary-gate branch consults `binding`: it resolves each of the
        applied operation's targets to a `BoundResource`, then uses the
        resulting device labels for the implementation-map lookup and the
        resulting engine indices everywhere a flat index is needed (shape
        checks, `ApplyMatrixStep`, noise selection). `binding` is optional so
        a caller resolving a plan directly (bypassing `run()`) does not have
        to construct one; when omitted, one is built via
        `_create_resource_binding`.

        Raises:
            UnsupportedResourceOperandError: If a unitary gate's target
                cannot be resolved by any binder in `binding` (e.g. a
                `RegisterView` reaching a backend that installs no binder
                able to claim it).
        """
        if binding is None:
            binding = self._create_resource_binding(program, layout)

        plan: list[ResolvedStep] = []
        has_measurement = False
        has_reset = False
        has_channel = False

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
                        confusions=self._resolve_confusions(measured_indices, layout),
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                if isinstance(step.operation, BarrierGate):
                    continue

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    target_indices = tuple(
                        layout.subsystem_index(t) for t in step.targets
                    )
                    # Reset-attached channels ("apply after the ideal reset")
                    # are designed but not wired yet; raising keeps the gap
                    # loud instead of silently dropping registered noise.
                    if self._noise_model.channels_for(
                        ResetGate, target_indices, layout
                    ):
                        raise UnsupportedOperationError(
                            "channel noise attached to Reset is not supported yet"
                        )
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(ResetStep(reset_indices=target_indices, condition=cond))
                    continue

                bound = tuple(binding.resolve(t, layout) for t in step.targets)
                device_labels = tuple(b.device_label for b in bound)
                engine_indices = tuple(b.engine_index for b in bound)

                rule = self._implementation_for(step.operation, device_labels)
                try:
                    matrix = rule(step.operation, targets=step.targets)
                except Exception as exc:
                    raise MatrixImplementationError(
                        f"implementation for {type(step.operation).__name__} raised: {exc}"
                    ) from exc

                # Check matrix shape matches target dimensions
                target_dims = tuple(layout.system_dims[i] for i in engine_indices)
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
                        matrix=matrix,
                        target_indices=engine_indices,
                        condition=cond,
                        # Identity, not mechanics: the backend forwards which
                        # implementation was selected; the engine alone decides
                        # what (if anything) that means for kernel choice.
                        kernel_key=rule._kernel_key(
                            step.operation, targets=step.targets
                        ),
                    )
                )

                # Attached channels resolve inline, mirroring the gate
                # path above: rule lookup, Kraus resolution, shape check,
                # step append - one ApplyChannelStep per channel, inheriting
                # the gate's condition. Noise selection stays in engine-index
                # space; device labels are exclusively for the
                # implementation-map lookup above.
                for channel in self._noise_model.channels_for(
                    type(step.operation), engine_indices, layout
                ):
                    has_channel = True
                    rule = self._channel_map.get(type(channel))
                    if rule is None:
                        raise UnsupportedOperationError(
                            f"{type(channel).__name__} has no channel "
                            "implementation on this backend"
                        )
                    kraus_ops = tuple(rule(channel, targets=step.targets))
                    _validate_kraus_shapes(kraus_ops, expected, type(channel).__name__)
                    plan.append(
                        ApplyChannelStep(
                            kraus_ops=kraus_ops,
                            target_indices=engine_indices,
                            condition=cond,
                        )
                    )

        return (
            plan,
            _PlanFacts(
                has_measurement=has_measurement,
                has_reset=has_reset,
                has_channel=has_channel,
            ),
        )

    def _resolve_confusions(
        self,
        measured_indices: tuple[int, ...],
        layout: FlatResourceLayout,
    ) -> tuple[Any, ...] | None:
        """Resolve per-subsystem readout confusion matrices for one measurement.

        ``readout_error_for`` is the single source of truth per subsystem;
        this method only collapses an all-``None`` resolution back to ``None``
        so the noise-free (and the common) case allocates nothing on the step.

        Raises :py:exc:`~fatqat.errors.BackendValidationError` if a selected
        matrix's dimension does not match the measured subsystem.
        """
        resolved = []
        for measured in measured_indices:
            confusion = self._noise_model.readout_error_for(measured, layout)
            if confusion is not None:
                dim = layout.system_dims[measured]
                if confusion.shape != (dim, dim):
                    raise BackendValidationError(
                        f"readout confusion matrix of shape {confusion.shape} "
                        f"selected for subsystem {measured} of dimension {dim}"
                    )
            resolved.append(confusion)
        if all(confusion is None for confusion in resolved):
            return None
        return tuple(resolved)

    def validate_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report which parts of a noise model this backend can execute.

        A channel descriptor type is supported exactly when the backend's
        channel implementation map has a rule for it - the map's coverage is
        the capability declaration. Non-empty ``qubit_noise`` is rejected as
        a structural mismatch (continuously-active per-subsystem noise is
        pulse-family territory), and Reset-keyed entries are rejected until
        reset-attached channels are wired.

        Args:
            noise_model: The noise model to check; it is not executed.

        Returns:
            A frozen report naming accepted and rejected sources.
        """
        accepted: list[str] = []
        rejected: list[str] = []
        warnings_: list[str] = []
        for channel_type in sorted(
            noise_model.channel_types(), key=lambda c: c.__name__
        ):
            if self._channel_map.get(channel_type) is None:
                rejected.append(channel_type.__name__)
                warnings_.append(
                    f"{channel_type.__name__} has no channel implementation "
                    "on this backend"
                )
            else:
                accepted.append(channel_type.__name__)
        if noise_model.has_readout_error():
            accepted.append("readout_error")
        if noise_model.qubit_noise:
            rejected.append("qubit_noise")
            warnings_.append(
                "qubit_noise holds continuously-active noise for pulse-family "
                "backends; the matrix family cannot consume it"
            )
        if noise_model.has_noise_for(ResetGate):
            rejected.append("Reset")
            warnings_.append("channel noise attached to Reset is not supported yet")
        return NoiseSupportReport(
            supported=not rejected,
            accepted_sources=tuple(accepted),
            rejected_sources=tuple(rejected),
            warnings=tuple(warnings_),
        )


def _resolve_result_request(
    config: Any,
    facts: _PlanFacts,
    request_cls: type,
    state_field: str,
    nonunitary_is_stochastic: bool,
) -> Any:
    """Resolve default result fields from config and lowered program facts.

    Counts default to measurement presence. The state field defaults to
    non-stochastic execution, where stochasticity is representation-dependent:
    measurement always; reset and channel noise only when
    ``nonunitary_is_stochastic`` (a statevector samples one branch of any
    non-unitary map; a density matrix applies it as a deterministic channel).
    """
    stochastic = facts.has_measurement or (
        nonunitary_is_stochastic and (facts.has_reset or facts.has_channel)
    )
    counts = config.counts if config.counts is not None else facts.has_measurement
    state = getattr(config, state_field)
    if state is None:
        state = not stochastic
    return request_cls(counts=counts, **{state_field: state})
