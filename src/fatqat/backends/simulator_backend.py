"""Unified matrix-family simulator backend with Qiskit-style method selection.

`SimulatorBackend` is the single entry point for matrix-family simulation:
``SimulatorBackend(method="statevector")`` and
``SimulatorBackend(method="density_matrix")`` (aliases ``"SV"`` / ``"DM"``,
case-insensitive) select the state representation, exactly like Qiskit Aer's
``AerSimulator(method=...)``. It is the only simulator backend:
per-representation backend classes do not exist.

Everything method-independent lives here once: per-run simulation/result-config
normalization, lowering (including the compiler-facing `Barrier` skip),
validation, execution orchestration, and public `Result` assembly. The
method-dependent facts are bound as instance attributes in ``__init__`` -
the backend never branches on method afterwards:

- ``_state_field``: the native state field name (``"statevector"`` or
  ``"density_matrix"``). Drives the result-config flag read, the `Result`
  keyword, the availability name, the metadata echo, and validation wording.
- ``_simulator_cls``: the `Simulator` subclass the (method, runtime) pair
  drives (`NumpySVSimulator`, `NumpyDMSimulator`, or the optional
  `NumbaSVSimulator` / `NumbaDMSimulator`); one instance is bound to
  ``_simulator`` and reused across runs.
- ``_request_cls``: the method's engine-request value object. The public
  ``final_state`` result request is translated to that representation's
  native state field immediately before execution.
- ``_nonunitary_is_stochastic``: whether non-unitary maps (reset, channel
  noise) make execution stochastic for the state representation (`True`
  for statevector, which must sample one branch of any non-unitary map;
  `False` for density matrix, which applies them as deterministic
  channels).

The backend/simulator seam: this class constructs the method's `Simulator`
subclass once, then calls ``simulator.initialize(system_dims, n_clbits)``
and ``simulator.run(plan, shots, seed, request, config=...) -> RawResult`` per run,
exactly as the per-method backends did against the engine before this
refactor.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import prod
from typing import Any

from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    UnsupportedOperationError,
)
from .._engine_index_allocation import _EngineIndexAllocation
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
from ..registers import RegisterRef
from ..resource_layout import DeviceOperand, ResourceLayout
from ..result import (
    Result,
    _ResultConfig,
    counts_dict_from_arrays,
)
from ..simulator import NumpyDMSimulator, NumpySVSimulator, Simulator
from .backend_utils import (
    _LoweringContext,
    _PlanFacts,
    _lower_measurement_boundary,
    _normalize_config,
    _resolve_condition,
)
from .engine_contract import (
    RawResult,
    _DensityMatrixResultRequest,
    _EngineConfig,
    _SimulationConfig,
    _StateVectorResultRequest,
)
from .view_normalization import ProgramInstruction, _break_grouped_operations
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


def _gate_implementation_for(
    operation: Operation, device_operands: DeviceOperands, impl_map: ImplementationMap
) -> MatrixImplementation:
    """Resolve the matrix rule for a gate operation on a device target key.

    Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` if the operation has no rule at
    all, or if it has rules but none for this target key — the message
    distinguishes the two.
    """
    if not impl_map.supports(operation):
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported by this backend"
        )
    rule = impl_map.implementation_for(operation, device_operands=device_operands)
    if rule is None:
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported on device operands {device_operands}"
        )
    return rule


def _lower_measurement(
    step: Measurement,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> MeasurementStep:
    """Lower one `Measurement` instruction into a `MeasurementStep`."""
    # Only used to build reported_digit_maps below; _lower_measurement_boundary
    # independently resolves the authoritative measured_indices from step.targets.
    digit_map_indices = tuple(
        engine_index_allocation.subsystem_index(q) for q in step.targets
    )
    reported_digit_maps = tuple(
        tuple(range(engine_index_allocation.system_dims[measured]))
        for measured in digit_map_indices
    )
    measured_indices, classical_indices, confusions = _lower_measurement_boundary(
        step,
        reported_digit_maps,
        resource_layout,
        engine_index_allocation,
        noise_model,
    )
    return MeasurementStep(
        measured_indices=measured_indices,
        classical_indices=classical_indices,
        confusions=confusions,
        # The identity map is the compatibility default (see steps.py); only
        # carry it explicitly when a confusion matrix needs it for the
        # reported-dimension check, so an identity, noise-free measurement
        # keeps the None default the numba fast path recognizes.
        reported_digit_maps=reported_digit_maps if confusions is not None else None,
    )


def _lower_reset(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> ResetStep:
    """Lower one `Reset` `AppliedOperation` into a `ResetStep`.

    Raises:
        UnsupportedOperationError: If channel noise is attached to Reset.
            Reset-attached channels ("apply after the ideal reset") are
            designed but not wired yet; raising keeps the gap loud instead
            of silently dropping registered noise. The selector lookup runs
            against logical/physical identity (never engine indices) before
            the guard raises.
    """
    target_indices = tuple(
        engine_index_allocation.subsystem_index(t) for t in step.targets
    )
    if noise_model.channels_for(ResetGate, step.targets, resource_layout):
        raise UnsupportedOperationError(
            "channel noise attached to Reset is not supported yet"
        )
    cond = _resolve_condition(step.condition, engine_index_allocation)
    return ResetStep(reset_indices=target_indices, condition=cond)


def _lower_gate(
    step: AppliedOperation,
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    impl_map: ImplementationMap,
    noise_model: NoiseModel,
    channel_map: ChannelImplementationMap,
) -> list[ResolvedStep]:
    """Lower one ordinary-gate `AppliedOperation`.

    Returns the gate's `ApplyMatrixStep` followed by one `ApplyChannelStep`
    per noise channel attached to this occurrence, in registration order,
    each inheriting the gate's condition. Every gate reaching here has only
    scalar targets: grouped expansion already happened before lowering, so
    each target maps to one device operand and one engine index. Targets
    were validated when this `AppliedOperation` was constructed.
    Implementation-map lookup uses device operands from the resource
    layout; every execution index comes from the engine index allocation
    instead.
    """
    device_operands = resource_layout.device_operands(step.targets)
    engine_indices = tuple(
        engine_index_allocation.subsystem_index(t) for t in step.targets
    )
    cond = _resolve_condition(step.condition, engine_index_allocation)

    rule = _gate_implementation_for(step.operation, device_operands, impl_map)
    try:
        matrix = rule(step.operation, targets=step.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(step.operation).__name__} raised: {exc}"
        ) from exc

    # Check matrix shape matches this instruction's target dims.
    target_dims = tuple(engine_index_allocation.system_dims[i] for i in engine_indices)
    expected = prod(target_dims)
    if matrix.shape != (expected, expected):
        raise BackendValidationError(
            f"{type(step.operation).__name__} resolved to a "
            f"{matrix.shape} matrix, incompatible with target "
            f"dimensions {target_dims} (expected "
            f"{(expected, expected)})"
        )

    steps: list[ResolvedStep] = [
        ApplyMatrixStep(
            matrix=matrix,
            target_indices=engine_indices,
            condition=cond,
            # Identity, not mechanics: the backend forwards which
            # implementation was selected; the engine alone decides
            # what (if anything) that means for kernel choice.
            kernel_key=rule._kernel_key(step.operation, targets=step.targets),
        )
    ]

    # Noise selection matches against the occurrence's logical targets
    # and/or resource-layout device operands (never engine indices);
    # engine indices are used only for the emitted ApplyChannelStep.
    for channel, extent in noise_model.channels_for(
        type(step.operation), step.targets, resource_layout
    ):
        channel_rule = channel_map.get(type(channel))
        if channel_rule is None:
            raise UnsupportedOperationError(
                f"{type(channel).__name__} has no channel "
                "implementation on this backend"
            )
        extent_indices = tuple(
            engine_index_allocation.subsystem_index(target) for target in extent
        )
        kraus_ops = tuple(channel_rule(channel, targets=extent))
        extent_dim = prod(
            engine_index_allocation.system_dims[index] for index in extent_indices
        )
        _validate_kraus_shapes(kraus_ops, extent_dim, type(channel).__name__)
        steps.append(
            ApplyChannelStep(
                kraus_ops=kraus_ops,
                target_indices=extent_indices,
                condition=cond,
            )
        )
    return steps


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

    Per-run ``simulation_config`` controls local execution only: ``seed``,
    ``max_workers``, and ``parallel_mode``. ``result_config`` controls the
    execution record: ``counts`` and ``final_state``. ``shots`` is an
    explicit ``run()`` argument, matching a hardware job's repetition count.

    The ``runtime`` argument selects the execution technology for the chosen
    representation - ``"numpy"`` (default) or ``"numba"`` (optional
    dependency). The runtime never changes
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
        ...     result_config={"counts": False, "final_state": True},
        ... ).result()
        >>> result.get_density_matrix()
        array([[0.5+0.j, 0.5+0.j],
               [0.5+0.j, 0.5+0.j]])
    """

    # Subclasses may replace this with a dataclass derived from
    # ``_ResultConfig`` to expose hardware-specific result artifacts. The
    # normalizer derives accepted keys from this schema, so unsupported fields
    # fail before execution.
    _result_config_cls: type[_ResultConfig] = _ResultConfig

    # Subclasses may replace this with a dataclass derived from
    # ``_SimulationConfig`` for hardware-specific simulator controls. The
    # base backend only consumes the inherited seed and engine-config portion.
    _simulation_config_cls: type[_SimulationConfig] = _SimulationConfig

    def __init__(
        self,
        method: str = "statevector",
        *,
        runtime: str = "numpy",
        implementation_map: ImplementationMap | None = None,
        noise: NoiseModel | None = None,
        channel_implementation_map: ChannelImplementationMap | None = None,
    ) -> None:
        """Create a simulator backend for the given method and runtime.

        Args:
            method: Simulation method: ``"statevector"`` or
                ``"density_matrix"``, or the case-insensitive short aliases
                ``"SV"`` / ``"DM"``.
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
                reproducibility contract. ``"numba"`` supports both methods
                and requires the optional ``numba`` dependency, which raises
                here, at construction, rather than at run time.
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
                of the supported names, or ``runtime="numba"`` is requested
                without the numba dependency installed.
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
            self._request_cls = _StateVectorResultRequest
            self._simulator_cls = NumpySVSimulator
            # A pure state cannot represent the mixed output of a non-unitary
            # map, so reset and channel noise must each sample one branch - a
            # random event, like measurement.
            self._nonunitary_is_stochastic = True
        else:
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
            try:
                # Lazy: numba is an optional dependency, and fatqat.simulator's
                # package __init__ deliberately never imports the nb module.
                from ..simulator.nb import NumbaDMSimulator, NumbaSVSimulator
            except ImportError as exc:
                raise BackendValidationError(
                    "runtime='numba' requires the optional numba dependency "
                    "(install the 'numba' group)"
                ) from exc
            self._simulator_cls = (
                NumbaSVSimulator if normalized == "statevector" else NumbaDMSimulator
            )
        self._runtime = normalized_runtime

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
        self._simulator = self._simulator_cls(config=_EngineConfig())
        self._simulator_system: tuple[tuple[int, ...], int] | None = None

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
        """Resolve this run's effective public resource layout.

        The base implementation is the generic simulator's trivial mapping
        policy: concatenate quantum registers in declaration order and assign
        device labels ``0, 1, ...``. A backend with predefined physical sites
        (or any other non-trivial mapping policy) overrides this hook; it is
        also where such a backend validates device-resource concerns like
        capacity, dimension, or grid fit, since those are properties of the
        logical-to-physical mapping, not of engine index allocation.

        Args:
            program: Program whose quantum registers should be mapped.

        Returns:
            The effective resource layout for this run.
        """
        labels: dict[RegisterRef, DeviceOperand] = {}
        index = 0
        for register in program.qreg:
            for i in range(register.size):
                labels[register[i]] = index
                index += 1
        return ResourceLayout(labels)

    def _allocate_engine_indices(self, program: Program) -> _EngineIndexAllocation:
        """Build this run's private engine-facing flat allocation.

        Args:
            program: Program whose registers should be flattened.

        Returns:
            Engine allocation mapping register references to flat subsystem
            and classical indices.
        """
        return _EngineIndexAllocation.from_program(program)

    def _lower_program(
        self,
        program: Program,
        *,
        context: _LoweringContext | None = None,
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Prepare and lower one program using the backend's resource policy.

        ``context`` lets a caller that already resolved this run's
        `ResourceLayout` and `_EngineIndexAllocation` (see ``run()``) thread both
        through unchanged, so lowering never re-resolves either. When omitted
        (standalone use, e.g. in tests), both are resolved once here.
        """
        if context is None:
            context = _LoweringContext(
                resource_layout=self._resolve_resource_layout(program),
                engine_index_allocation=self._allocate_engine_indices(program),
            )
        operations = _break_grouped_operations(program.operations)
        return self._lower(operations, context)

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job:
        """Validate, execute, and package one program run.

        Resolves the program's effective resource layout and private engine
        index allocation, chooses an execution strategy, runs the circuit, and
        returns an eager ``Job`` whose ``result()`` yields a ``Result``.

        ``simulation_config`` controls local execution only: ``seed``,
        ``max_workers``, and ``parallel_mode``. ``result_config`` describes
        the requested result artifacts: ``counts`` and ``final_state``. The
        latter asks a simulator to return its terminal state in the
        representation selected by this backend's ``method``.

        Args:
            program: Program to execute.
            shots: Number of circuit repetitions. Counts and a stochastic
                final-state request require a positive integer; a
                non-stochastic final-state-only request ignores it.
            simulation_config: Optional simulator-only dictionary. Unknown or
                incompatible entries raise an error.
            result_config: Optional result-request dictionary. Unknown or
                incompatible entries raise an error.

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
        simulation = _normalize_config(
            simulation_config,
            self._simulation_config_cls,
            "simulation_config",
            backend_name=type(self).__name__,
        )
        config = _normalize_config(
            result_config,
            self._result_config_cls,
            "result_config",
            backend_name=type(self).__name__,
        )
        # Both hooks are resolved exactly once per run, on the direct-raise
        # validation path, before the execution try block below: capacity,
        # dimension, grid-fit, and mapping failures must raise directly from
        # run(), never become a failed Job. The resource layout is the
        # public-facing effective mapping (available to backend validation);
        # the engine index allocation stays private to execution preparation. Both
        # are paired into one private lowering context and threaded through
        # `_lower_program`/`_lower` unchanged, so lowering never re-resolves
        # either value.
        resource_layout = self._resolve_resource_layout(program)
        engine_index_allocation = self._allocate_engine_indices(program)
        # Strict selector-identity validation runs immediately after the
        # effective resource layout is known and before any lowering/plan
        # step is built, on this same direct-raise path: a foreign ref or
        # unmapped device label fails run() directly rather than being
        # silently skipped in channels_for()/readout_error_for() matching.
        self._noise_model.validate_for(program, resource_layout)
        context = _LoweringContext(
            resource_layout=resource_layout,
            engine_index_allocation=engine_index_allocation,
        )
        plan, facts = self._lower_program(program, context=context)
        self._validate(config, shots, facts)
        self._validate_additional_config(
            config=config,
            simulation=simulation,
            shots=shots,
            facts=facts,
        )
        try:
            return Job.done(
                self._execute(
                    config,
                    simulation,
                    shots,
                    plan,
                    facts,
                    engine_index_allocation.system_dims,
                    engine_index_allocation.classical_dims,
                    engine_index_allocation.n_clbits,
                )
            )
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(self, config: _ResultConfig, shots: int, facts: _PlanFacts) -> None:
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
        requested_state = config.final_state is True

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

    def _validate_additional_config(
        self,
        *,
        config: _ResultConfig,
        simulation: _SimulationConfig,
        shots: int,
        facts: _PlanFacts,
    ) -> None:
        """Validate backend-specific configuration against this program run.

        Subclasses use this pre-execution hook for constraints that depend on
        the result request, simulation controls, shot count, or lowered
        program facts. Raise ``BackendValidationError`` with a user-facing
        explanation when a declared configuration is incompatible.
        """

    # --- execution ---
    def _execute(
        self,
        config: _ResultConfig,
        simulation: _SimulationConfig,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        system_dims: tuple[int, ...],
        classical_dims: tuple[int, ...],
        n_clbits: int,
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

        raw = self._simulator.run(
            plan,
            shots,
            simulation.seed,
            request,
            config=simulation.engine_config(),
        )
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

        extra_data = self._additional_result_data(
            config=config,
            simulation=simulation,
            raw=raw,
        )
        return Result(
            counts=counts,
            available=frozenset(available),
            classical_dims=classical_dims,
            data=extra_data,
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "method": self._state_field,
                "runtime": self._runtime,
                "simulation_config": asdict(simulation),
                "result_config": asdict(config),
            },
            **{self._state_field: state},
        )

    def _additional_result_data(
        self,
        *,
        config: _ResultConfig,
        simulation: _SimulationConfig,
        raw: RawResult,
    ) -> Mapping[str, Any]:
        """Return backend-specific result artifacts for this completed run.

        A hardware subclass extends the corresponding config dataclass and
        overrides this hook to turn its requested fields into public result
        data. The common backend has no extra artifacts.
        """
        return {}

    def _lower(
        self,
        operations: Sequence[ProgramInstruction],
        context: _LoweringContext,
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Dispatches each instruction to a per-type free function
        (`_lower_measurement`/`_lower_reset`/`_lower_gate`), threading this
        backend's noise model / implementation map / channel map through
        explicitly - none of the three is overridden by any backend today,
        so they take their dependencies as plain parameters instead of
        reading `self`. `Barrier` is recognized by type and skipped
        entirely here - it is a compiler-facing marker with no simulation
        semantics, so it emits no step and cannot affect execution strategy
        or result defaults. `_PlanFacts` is derived from the finished
        `plan` rather than tracked by mutation, so it can never drift from
        what the plan actually contains.

        The caller supplies a scalar-only instruction stream and the run's
        private lowering context. `context.resource_layout` is used for
        `ImplementationMap` lookup (`device_operands`) and for
        `NoiseModel.channels_for()` physical-selector matching (against the
        occurrence's logical target refs); `context.engine_index_allocation` is
        used for every execution index/dimension - `ApplyMatrixStep`/
        `MeasurementStep`/`ResetStep` targets and conditions. Grouped
        frontend operations are expanded before this method is called.
        """
        resource_layout = context.resource_layout
        engine_index_allocation = context.engine_index_allocation
        plan: list[ResolvedStep] = []

        for step in operations:
            if isinstance(step, Measurement):
                plan.append(
                    _lower_measurement(
                        step,
                        resource_layout,
                        engine_index_allocation,
                        self._noise_model,
                    )
                )
            elif isinstance(step, AppliedOperation):
                if isinstance(step.operation, BarrierGate):
                    continue
                if isinstance(step.operation, ResetGate):
                    plan.append(
                        _lower_reset(
                            step,
                            resource_layout,
                            engine_index_allocation,
                            self._noise_model,
                        )
                    )
                else:
                    plan.extend(
                        _lower_gate(
                            step,
                            resource_layout,
                            engine_index_allocation,
                            self._impl_map,
                            self._noise_model,
                            self._channel_map,
                        )
                    )

        return (
            plan,
            _PlanFacts(
                has_measurement=any(isinstance(s, MeasurementStep) for s in plan),
                has_reset=any(isinstance(s, ResetStep) for s in plan),
                has_channel=any(isinstance(s, ApplyChannelStep) for s in plan),
            ),
        )

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
        for source_type in sorted(
            noise_model.continuous_noise_types(), key=lambda source: source.__name__
        ):
            rejected.append(source_type.__name__)
            warnings_.append(
                f"{source_type.__name__} is continuously active pulse-family "
                "noise and is not supported by this matrix backend"
            )
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
    config: _ResultConfig,
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
    state = config.final_state
    if state is None:
        state = not stochastic
    return request_cls(counts=counts, **{state_field: state})
