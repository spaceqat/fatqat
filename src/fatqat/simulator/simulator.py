"""Unified gate-level simulator backend with Qiskit-style method selection.

`Simulator` is the single entry point for matrix-family simulation:
``Simulator(method=...)`` selects the representation - ``"statevector"`` /
``"density_matrix"`` (aliases ``"SV"`` / ``"DM"``) for a *state*, or
``"unitary"`` / ``"superop"`` for the program's *map* - exactly like Qiskit
Aer's ``AerSimulator(method=...)``. Names are case-insensitive. It is the only
gate-level backend: per-representation backend classes do not exist.
Pulse-level emulation is the sibling package :mod:`fatqat.emulator`.

Everything method-independent lives here once: per-run simulation/result-config
normalization, lowering (including the compiler-facing `Barrier` skip),
validation, execution orchestration, and public `Result` assembly. The
method-dependent facts come from one `_MethodSpec` table entry, bound as
instance attributes in ``__init__`` - the backend never branches on method
afterwards:

- ``_state_field``: the native result field name, which is the canonical
  method name itself. Drives the result-config flag read, the `Result`
  keyword, the availability name, the metadata echo, and validation wording.
- ``_engine_cls``: the `MatrixEngine` subclass the (method, runtime) pair
  drives (`NumpySVEngine`, `NumpyDMEngine`, `NumpyUnitaryEngine`,
  `NumpySuperopEngine`, or their Numba twins); one instance is bound
  to ``_engine`` and reused across runs.
- ``_request_cls``: the method's engine-request value object. The public
  ``final_state`` result request is translated to that representation's
  native state field immediately before execution.
- ``_nonunitary_is_stochastic``: whether non-unitary maps (reset, channel
  noise) make the final state stochastic. Statevector samples one branch;
  unitary keeps the defensive `True` fact but rejects non-unitary maps through
  ``_executes_nonunitary``; density-matrix and superoperator methods apply
  deterministic channels and use `False`.
- ``_is_operator`` / ``_executes_nonunitary``: what the representation can
  execute at all, checked in `_validate_method_support` before any run.

The backend/engine seam: this class constructs the method's `MatrixEngine`
subclass once. Each run analyzes one frozen plan, resolves static engine
capabilities and public controls into one private policy, then copies only the
runtime semantic shape into a slim execution context. The engine materializes
its numeric payload once and performs local numeric execution; this module owns
process dispatch and public result assembly.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .._parameter_binding import (
    _normalize_parameter_batch,
    _raise_for_unbound_parameters,
)
from ..errors import BackendValidationError
from .._index_allocation import (
    _ClassicalAllocation,
    _EngineAllocation,
    _describe_state_axes,
)
from ..implementation import MatrixImplementationMap, default_matrix_implementation_map
from ..job import Job
from ..noise import (
    AmplitudeDamping,
    ChannelImplementationMap,
    Depolarizing,
    PhaseDamping,
    NoiseModel,
    NoiseSupportReport,
    Loss,
    ThermalRelaxation,
    default_channel_implementation_map,
)
from ..operations import BarrierGate, Measurement, ResetGate
from ..parameters import Parameter, ParameterVector
from ..program import AppliedOperation, Program
from ..registers import RegisterRef
from ..resource_layout import DeviceOperand, ResourceLayout
from ..result import (
    Result,
    _ResultConfig,
    counts_dict_from_arrays,
    reduce_to_counts,
)
from ._engine.base import MatrixEngine
from ._engine.parallel import _run_shots_in_processes
from ._engine.np import (
    NumpyDMEngine,
    NumpySuperopEngine,
    NumpySVEngine,
    NumpyUnitaryEngine,
)
from .._backends.backend_utils import (
    _LoweringContext,
    _normalize_config,
    _resolve_result_flags,
    _validate_result_shots,
)
from . import planning
from ._execution_contract import _ExecutionContext, _ExecutionPolicy, _PlanFacts
from .._backends.engine_contract import (
    RawResult,
    _DensityMatrixResultRequest,
    _ResultRequest,
    _SimulationConfig,
    _StateVectorResultRequest,
    _SuperopResultRequest,
    _UnitaryResultRequest,
)
from ._execution_policy import (
    _materialization_policy,
    _resolve_execution_policy,
    _should_probe_compiled_multi_shot,
    _validate_execution_controls,
)
from .._backends.view_normalization import ProgramInstruction, _break_grouped_operations
from .._backends.steps import (
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
    "unitary": "unitary",
    "superop": "superop",
}


def _dispatch_execution(
    engine: MatrixEngine,
    context: _ExecutionContext,
    payload: Any,
    policy: _ExecutionPolicy,
) -> RawResult:
    """Dispatch one prepared execution without leaking routes into engines."""
    state_requested = any(
        getattr(context.request, field, False)
        for field in ("statevector", "density_matrix", "unitary", "superop")
    )
    if policy.use_compiled_multi_shot_kernel:
        assert context.execution_shape == "per_shot"
        assert context.request.counts and not state_requested
        assert context.initial_occupied is None

    if policy.shot_strategy in ("none", "serial", "threads"):
        return engine.execute_local(context, payload, policy)

    assert policy.use_compiled_multi_shot_kernel is False
    assert policy.shot_strategy == "processes"
    assert context.execution_shape == "per_shot"
    assert context.request.counts and not state_requested
    snapshots = _run_shots_in_processes(type(engine), context, payload, policy)
    rows = np.asarray(snapshots, dtype=int).reshape((len(snapshots), context.n_clbits))
    outcome_keys, outcome_counts = reduce_to_counts(rows)
    return RawResult(
        outcome_keys=outcome_keys,
        outcome_counts=outcome_counts,
    )


@dataclass(frozen=True)
class _MethodSpec:
    """Everything the chosen simulation method binds into a `Simulator`.

    Attributes:
        request_cls: The method's engine-request value object.
        numpy_engine: The `MatrixEngine` subclass for ``runtime="numpy"``.
        numba_engine_name: The `fatqat.simulator._engine.nb` attribute naming
            the ``runtime="numba"`` twin, held as a name so the module is
            resolved lazily.
        nonunitary_is_stochastic: Whether non-unitary maps (reset, channel
            noise) make execution stochastic for this representation.
        is_operator: Whether the method computes the program's map rather than
            a state under it.
        executes_nonunitary: Whether the representation can apply a non-unitary
            map at all.
    """

    request_cls: type
    numpy_engine: type[MatrixEngine]
    numba_engine_name: str
    nonunitary_is_stochastic: bool
    is_operator: bool
    executes_nonunitary: bool


_METHOD_SPECS: dict[str, _MethodSpec] = {
    "statevector": _MethodSpec(
        request_cls=_StateVectorResultRequest,
        numpy_engine=NumpySVEngine,
        numba_engine_name="NumbaSVEngine",
        # nonunitary_is_stochastic: channel and reset
        # A pure state must sample one branch of any non-unitary map.
        nonunitary_is_stochastic=True,
        is_operator=False,
        executes_nonunitary=True,
    ),
    "density_matrix": _MethodSpec(
        request_cls=_DensityMatrixResultRequest,
        numpy_engine=NumpyDMEngine,
        numba_engine_name="NumbaDMEngine",
        # A density matrix holds the full ensemble, so only measurement is
        # stochastic.
        nonunitary_is_stochastic=False,
        is_operator=False,
        executes_nonunitary=True,
    ),
    "unitary": _MethodSpec(
        request_cls=_UnitaryResultRequest,
        numpy_engine=NumpyUnitaryEngine,
        numba_engine_name="NumbaUnitaryEngine",
        nonunitary_is_stochastic=True,  # reject by validation, not implemented.
        is_operator=True,
        executes_nonunitary=False,
    ),
    "superop": _MethodSpec(
        request_cls=_SuperopResultRequest,
        numpy_engine=NumpySuperopEngine,
        numba_engine_name="NumbaSuperopEngine",
        nonunitary_is_stochastic=False,
        is_operator=True,
        executes_nonunitary=True,
    ),
}


class Simulator:
    """Matrix-family simulator backend for ``fatqat.Program`` execution.

    The simulation method selects the representation and its semantics;
    everything else (supported operations, grouped measurement, feedforward
    conditions, reset, execution strategies, result handling) is
    method-independent.

    Two methods simulate a **state** the program prepares:

    - ``method="statevector"`` (alias ``"SV"``): pure-state simulation. The
      native result field is ``statevector``. Reset samples a branch, so any
      reset makes execution stochastic and forces per-shot replay.
    - ``method="density_matrix"`` (alias ``"DM"``): exact mixed-state
      simulation. The native result field is ``density_matrix``. Reset is
      the deterministic partial-trace channel, so reset alone neither makes
      a program stochastic nor forces per-shot execution.

    Two methods compute the program's **map** instead, in one deterministic
    pass with no shots and no sampling. Both reject measurement, feedforward
    conditions, and a ``counts`` request, since an operator has no classical
    register and no outcomes to sample:

    - ``method="unitary"``: the program's ``(D, D)`` unitary, where ``D`` is
      the product of the subsystem dimensions. Native result field
      ``unitary``. Reset and channel noise are rejected - a unitary cannot
      represent a non-unitary map.
    - ``method="superop"``: the program's ``(D**2, D**2)`` super-operator using
      column-stacking vectorization of density matrices
      (``rho.reshape(-1, order="F")``). This describes the mathematical
      vectorization, not the NumPy memory layout of the returned matrix. Native
      result field ``superop``. Reset and channel noise are accepted; both are
      exact channels, as under density-matrix semantics. Memory grows as
      ``16**n`` for ``n`` qubits, so this method is for small circuits.

    A state run is classified into a fast path (evolve once, sample requested
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

    Per-run ``simulation_config`` accepts ``seed``, ``shot_parallelism``,
    ``kernel_parallelism``, ``max_workers``, and ``fusion``. Shot parallelism
    controls independent trajectories; kernel parallelism controls numerical
    work inside one evolution and never executes program operations out of
    order. Both axes default to ``"auto"`` and FATQAT resolves at most one
    parallel axis. ``max_workers`` is a shared concurrency ceiling, not a
    promise that every worker receives useful work. ``fusion`` is independent
    of concurrency and defaults off.
    ``result_config`` controls the execution record: ``counts`` and
    ``final_state``. ``shots`` is an explicit ``run()`` argument, matching a
    hardware job's repetition count.

    The ``runtime`` argument selects the execution technology for the chosen
    representation - ``"numba"`` (default) or ``"numpy"``. The runtime keeps
    the same simulation and numerical-tolerance contract while changing how
    the work is executed;
    dynamic-shot worker processes use the selected runtime as well. The Numba
    runtime selects one internal thread axis appropriate to the
    chosen axis; process workers are confined to one Numba thread so process
    and thread parallelism never nest. A caller that needs predictable latency
    for a small one-off run may select NumPy with both parallelism axes set to
    ``"serial"``.

    A backend instance reuses one simulator across runs, so it is efficient for
    repeated calls from one caller but is not safe for concurrent ``run()``
    calls.

    Examples:
        Density-matrix simulation, Qiskit style:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(1)
        >>> program.add(ops.H, 0)
        >>> result = fq.simulator.Simulator(method="DM").run(
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
    # base backend consumes the inherited matrix-execution fields.
    _simulation_config_cls: type[_SimulationConfig] = _SimulationConfig

    # Whether this backend implements the per-shot atom-occupancy lifecycle
    # Loss needs (loading, per-shot loss, refill). False on the generic
    # matrix backends, which reject Loss via check_noise_support rather than
    # silently ignoring it; AtomArraySimulator sets it True.
    _supports_loss: bool = False

    def __init__(
        self,
        method: str = "statevector",
        *,
        runtime: str = "numba",
        implementation_map: MatrixImplementationMap | None = None,
        noise: NoiseModel | None = None,
        channel_implementation_map: ChannelImplementationMap | None = None,
    ) -> None:
        """Create a simulator backend for the given method and runtime.

        Args:
            method: Simulation method: ``"statevector"`` or
                ``"density_matrix"`` (case-insensitive short aliases ``"SV"``
                / ``"DM"``) for a state, or ``"unitary"`` / ``"superop"`` for
                the program's map.
            implementation_map: Optional matrix implementation map controlling
                which operations this backend supports and how their matrices
                are built. ``None`` (the default) uses
                ``default_matrix_implementation_map()``. The backend copies
                whatever map it receives, so mutating the caller's map object
                after construction does not change this backend's behavior.
            runtime: Execution technology: ``"numba"`` (the default) or
                ``"numpy"``, case-insensitive. The runtime selects *how* the
                chosen state representation is computed, never its semantics:
                results are identical up to the documented per-simulator RNG
                reproducibility contract. Both runtimes support every method.
            noise: Optional :py:class:`~fatqat.NoiseModel` applied to every
                run. ``None`` means noise-free execution. The backend captures
                the model's registrations once at construction.
            channel_implementation_map: Optional map controlling which
                `Channel` descriptor types this backend can resolve and how
                their Kraus operators are built. ``None`` (the default) uses
                ``default_channel_implementation_map()``. Copied, like
                ``implementation_map``.

        Raises:
            BackendValidationError: If ``method`` or ``runtime`` is not one
                of the supported names, the required numba dependency cannot
                be imported, or the captured noise model contains a source
                this backend cannot execute.
        """
        normalized = _METHOD_ALIASES.get(str(method).lower())
        if normalized is None:
            raise BackendValidationError(
                f"unsupported method={method!r}; expected one of "
                "'statevector'/'SV', 'density_matrix'/'DM', 'unitary', or "
                "'superop'"
            )
        normalized_runtime = str(runtime).lower()
        if normalized_runtime not in ("numpy", "numba"):
            raise BackendValidationError(
                f"unsupported runtime={runtime!r}; expected 'numpy' or 'numba'"
            )
        # The single dispatch point; the canonical method name doubles as the
        # native result field name.
        spec = _METHOD_SPECS[normalized]
        self._state_field = normalized
        self._request_cls = spec.request_cls
        self._nonunitary_is_stochastic = spec.nonunitary_is_stochastic
        self._is_operator = spec.is_operator
        self._executes_nonunitary = spec.executes_nonunitary
        self._engine_cls: type[MatrixEngine] = spec.numpy_engine
        if normalized_runtime == "numba":
            try:
                # Lazy: fatqat.simulator's package __init__ deliberately never
                # imports the Numba engine module.
                from ._engine import nb
            except ImportError as exc:
                raise BackendValidationError(
                    "runtime='numba' requires the numba dependency; reinstall "
                    "fatqat to repair the environment"
                ) from exc
            self._engine_cls = getattr(nb, spec.numba_engine_name)
        self._runtime = normalized_runtime

        if implementation_map is None:
            implementation_map = default_matrix_implementation_map()
        self._impl_map = implementation_map.copy()
        if noise is not None and not isinstance(noise, NoiseModel):
            raise BackendValidationError("noise must be a NoiseModel or None")
        source_noise = noise if noise is not None else NoiseModel()
        self._noise_model = source_noise._copy()
        if channel_implementation_map is None:
            channel_implementation_map = default_channel_implementation_map()
        self._channel_map = channel_implementation_map.copy()
        report = self.check_noise_support(self._noise_model)
        if not report.supported:
            raise BackendValidationError("; ".join(report.warnings))
        # The engine object and its dimension-dependent numeric caches are
        # reused, while every execution allocates or resets evolving state.
        # A backend instance is not safe for concurrent run() calls.
        self._engine = self._engine_cls()

    @property
    def method(self) -> str:
        """Canonical name of the state representation this backend runs.

        One of ``"statevector"``, ``"density_matrix"``, ``"unitary"`` or
        ``"superop"`` - the canonical name, not whatever alias was passed in,
        so ``method="SV"`` reports ``"statevector"``. This is the same string
        that appears as ``result.metadata["method"]`` and as the result's
        native state field, so a caller can match on one value everywhere.

        Reading it does not run anything: the method is fixed at construction.
        That is what makes it usable as a precondition - a tool that only
        supports some representations can say so before paying for an
        evolution, instead of failing on a missing result field afterwards.

        Examples:
            >>> import fatqat as fq
            >>> fq.simulator.Simulator(method="SV").method
            'statevector'
            >>> fq.simulator.Simulator(method="DM").method
            'density_matrix'
        """
        return self._state_field

    def _default_resource_layout(self, program: Program) -> ResourceLayout:
        """Resolve this run's effective public resource layout.

        The base implementation is the generic simulator's trivial mapping
        policy: concatenate quantum registers in declaration order and assign
        device labels ``0, 1, ...``. A backend with predefined physical sites
        (or any other non-trivial mapping policy) overrides this hook; it is
        also where such a backend validates device-resource concerns like
        capacity, dimension, or grid fit, since those are properties of the
        program-to-device mapping, not of engine index allocation.

        Args:
            program: Program whose quantum registers should be mapped.

        Returns:
            The effective resource layout for this run.
        """
        labels: dict[RegisterRef, DeviceOperand] = {}
        index = 0
        for register in program.quantum_registers:
            for i in range(register.size):
                labels[register[i]] = index
                index += 1
        return ResourceLayout(labels)

    def _legal_device_operands(
        self, program: Program, resource_layout: ResourceLayout
    ) -> frozenset[DeviceOperand]:
        """Return the device operands legal for physical selectors this run."""
        return resource_layout.device_labels

    def _physical_dimension(
        self, device_operand: DeviceOperand, resource_layout: ResourceLayout
    ) -> int:
        """Return the local model dimension for one selected device operand."""
        return resource_layout._ref_for_label(device_operand).register.dim

    def _validate_resource_layout(
        self,
        program: Program,
        resource_layout: ResourceLayout,
        legal_device_operands: frozenset[DeviceOperand],
    ) -> None:
        """Validate one public layout without assigning numerical axes."""
        program_refs = frozenset(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        foreign = resource_layout.refs - program_refs
        if foreign:
            raise BackendValidationError(
                "resource layout contains a RegisterRef outside this program"
            )
        missing = program_refs - resource_layout.refs
        if missing:
            raise BackendValidationError(
                "resource layout does not cover every declared quantum ref"
            )
        if not resource_layout.device_labels <= legal_device_operands:
            raise BackendValidationError(
                "resource layout names a device operand outside this backend"
            )
        if len(resource_layout.device_labels) != len(resource_layout.refs):
            raise BackendValidationError(
                "resource layout maps multiple refs to one exclusive device operand"
            )
        for ref in resource_layout.refs:
            operand = resource_layout.device_label(ref)
            if ref.register.dim != self._physical_dimension(operand, resource_layout):
                raise BackendValidationError(
                    "resource layout maps incompatible program and device dimensions"
                )

    def _resolve_resource_layout(
        self,
        program: Program,
        supplied_layout: ResourceLayout | None = None,
    ) -> ResourceLayout:
        """Resolve and validate the effective public resource layout once."""
        resource_layout = (
            self._default_resource_layout(program)
            if supplied_layout is None
            else supplied_layout
        )
        legal = self._legal_device_operands(program, resource_layout)
        self._validate_resource_layout(program, resource_layout, legal)
        return resource_layout

    def _modeled_subsystems(
        self, program: Program, resource_layout: ResourceLayout
    ) -> tuple[tuple[DeviceOperand, int], ...]:
        """Return selected matrix subsystems in program declaration order."""
        return tuple(
            (
                resource_layout.device_label(ref),
                self._physical_dimension(
                    resource_layout.device_label(ref), resource_layout
                ),
            )
            for register in program.quantum_registers
            for ref in (register[index] for index in range(register.size))
        )

    def _allocate_engine_indices(
        self, program: Program, resource_layout: ResourceLayout
    ) -> _EngineAllocation:
        """Build the private engine allocation from the modeled physical order."""
        modeled = self._modeled_subsystems(program, resource_layout)
        return _EngineAllocation(
            tuple(operand for operand, _dimension in modeled),
            tuple(dimension for _operand, dimension in modeled),
        )

    def _lower_program(
        self,
        program: Program,
        *,
        context: _LoweringContext | None = None,
    ) -> tuple[tuple[ResolvedStep, ...], _PlanFacts]:
        """Prepare and lower one program using the backend's resource policy.

        ``context`` lets a caller that already resolved this run's
        `ResourceLayout` and `_EngineAllocation` (see ``run()``) thread both
        through unchanged, so lowering never re-resolves either. When omitted
        (standalone use, e.g. in tests), both are resolved once here.
        """
        plan, facts, _initial_occupied = self._prepare_program(program, context=context)
        return plan, facts

    def _prepare_program(
        self,
        program: Program,
        *,
        context: _LoweringContext | None = None,
    ) -> tuple[
        tuple[ResolvedStep, ...],
        _PlanFacts,
        frozenset[int] | None,
    ]:
        """Lower and freeze once, then derive common facts and occupancy."""
        if context is None:
            resource_layout = self._resolve_resource_layout(program)
            context = _LoweringContext(
                resource_layout=resource_layout,
                engine_allocation=self._allocate_engine_indices(
                    program, resource_layout
                ),
                classical_allocation=_ClassicalAllocation.from_program(program),
            )
        operations = _break_grouped_operations(program.operations)
        plan = tuple(self._lower(operations, context))
        facts, initial_occupied = self._analyze_lowered_plan(plan)
        return plan, facts, initial_occupied

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        resource_layout: ResourceLayout | None = None,
        initial_state: Any = None,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job[Result]:
        """Validate, execute, and package one program run.

        Resolves the program's effective resource layout and private engine
        index allocation, chooses an execution strategy, runs the circuit, and
        returns an eager ``Job`` whose ``result()`` yields a ``Result``.

        ``simulation_config`` accepts ``seed``, ``shot_parallelism``
        (``"auto"``, ``"serial"``, ``"threads"``, or ``"processes"``),
        ``kernel_parallelism`` (``"auto"``, ``"serial"``, or ``"threads"``),
        ``max_workers``, and ``fusion``. Kernel parallelism is numerical work
        inside one evolution; it never executes program operations out of
        order. FATQAT never nests the two parallel axes. Fusion defaults off
        and can be enabled only for the Numba density-matrix, unitary, and
        superoperator methods. ``result_config`` describes the
        requested result artifacts: ``counts`` and ``final_state``. The
        latter asks a simulator to return its terminal state in the
        representation selected by this backend's ``method``.

        Args:
            program: Program to execute.
            shots: Number of circuit repetitions. Counts and a stochastic
                final-state request require a positive integer; a
                non-stochastic final-state-only request ignores it.
            resource_layout: Optional public mapping from program quantum
                refs to backend device operands. A supplied layout must cover
                every declared quantum ref. The backend supplies its default
                mapping when omitted.
            initial_state: Optional state to start every shot from instead of
                the all-zero computational state. A statevector run takes a
                ``(D,)`` vector; a density-matrix run takes ``(D, D)``, or a
                ``(D,)`` vector read as the pure state it describes. ``D`` is
                the product of the program's subsystem dimensions, so qudits
                work without a separate form. Only the shape is checked - see
                :py:meth:`_validate_initial_state`.
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
        _raise_for_unbound_parameters(program.operations)
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
        capabilities = self._engine.capabilities
        _validate_execution_controls(simulation, capabilities)
        # Both hooks are resolved exactly once per run, on the direct-raise
        # validation path, before the execution try block below: capacity,
        # dimension, grid-fit, and mapping failures must raise directly from
        # run(), never become a failed Job. The resource layout is the
        # public-facing effective mapping (available to backend validation);
        # the engine index allocation stays private to execution preparation. Both
        # are paired into one private lowering context and threaded through
        # preparation/lowering unchanged, so lowering never re-resolves either
        # value.
        resource_layout = self._resolve_resource_layout(program, resource_layout)
        engine_allocation = self._allocate_engine_indices(program, resource_layout)
        classical_allocation = _ClassicalAllocation.from_program(program)
        initial_state = self._validate_initial_state(
            initial_state, engine_allocation.system_dims
        )
        # Strict selector-identity validation runs immediately after the
        # effective resource layout is known and before any lowering/plan
        # step is built, on this same direct-raise path: a foreign ref or
        # unmapped device label fails run() directly rather than being
        # silently skipped in selector matching.
        self._noise_model._validate_for(
            program, self._legal_device_operands(program, resource_layout)
        )
        lowering = _LoweringContext(
            resource_layout=resource_layout,
            engine_allocation=engine_allocation,
            classical_allocation=classical_allocation,
        )
        plan, facts, initial_occupied = self._prepare_program(program, context=lowering)
        request = self._validate(
            config,
            shots,
            facts,
            initial_occupied=initial_occupied,
        )
        self._validate_additional_config(
            config=config,
            simulation=simulation,
            shots=shots,
            facts=facts,
        )
        counts_requested = request.counts
        state_requested = getattr(request, self._state_field)
        compiled_multi_shot_compatible = False
        if _should_probe_compiled_multi_shot(
            simulation,
            facts=facts,
            counts_requested=counts_requested,
            state_requested=state_requested,
            initial_occupied=initial_occupied,
        ):
            compiled_multi_shot_compatible = (
                self._engine.compiled_multi_shot_compatible(plan)
            )
        policy = _resolve_execution_policy(
            simulation,
            facts=facts,
            counts_requested=counts_requested,
            state_requested=state_requested,
            capabilities=capabilities,
            compiled_multi_shot_compatible=compiled_multi_shot_compatible,
            shots=shots,
            initial_occupied=initial_occupied,
            plan_is_empty=not plan,
        )
        execution = _ExecutionContext(
            execution_shape=facts.execution_shape,
            request=request,
            system_dims=tuple(engine_allocation.system_dims),
            n_clbits=classical_allocation.n_clbits,
            shots=shots,
            seed=simulation.seed,
            initial_state=initial_state,
            initial_occupied=initial_occupied,
        )
        deferred_measurements = facts.deferred_measurements
        written_clbits = facts.written_clbits
        try:
            raw = self._execute_engine(
                plan=plan,
                deferred_measurements=deferred_measurements,
                context=execution,
                policy=policy,
            )
            result = self._assemble_result(
                raw=raw,
                config=config,
                simulation=simulation,
                lowering=lowering,
                written_clbits=written_clbits,
                request=request,
                shots=shots,
            )
            return Job(status="DONE", result=result)
        except Exception as exc:  # execution-stage failure
            return Job(status="ERROR", error=exc)

    def run_sweep(
        self,
        program: Program,
        bindings: Mapping[Parameter | ParameterVector, object],
        *,
        shots: int = 1024,
        resource_layout: ResourceLayout | None = None,
        initial_state: Any = None,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job[list[Result]]:
        """Bind and execute every row of one complete parameter batch.

        Single parameters accept shape ``(N,)`` and vectors accept shape
        ``(N, M)``. The returned eager job contains one ordinary ``Result`` per
        row, in input order. Version 1 delegates to :meth:`run` once per row.

        Args:
            program: Parameterized template program.
            bindings: Complete object-keyed parameter batch.
            shots: Number of repetitions forwarded to every row.
            resource_layout: Optional layout forwarded unchanged to every row.
            initial_state: Optional starting state forwarded unchanged to every row.
            simulation_config: Simulator options forwarded unchanged. An
                explicit seed is reused for every row, so sampled row errors
                are correlated; see :doc:`../guide/parameters-and-sweeps`.
            result_config: Result request forwarded unchanged.

        Returns:
            An eager job carrying an ordered list of row results. If a point
            job fails, ``result()`` re-raises that error and no partial result
            list is exposed.

        Raises:
            TypeError: If ``bindings`` is not an object-keyed mapping or a
                batch contains values other than built-in ``int``/``float``
                or NumPy integer/floating scalars.
            ValueError: If the program is not parameterized, assignments are
                missing or duplicated, or batch ranks and lengths disagree.
            BackendValidationError: If a bound row or forwarded run option
                fails normal Simulator validation.

        Examples:
            Sweep one angle and request the final state from every row:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> theta = fq.Parameter("theta")
            >>> program = fq.Program(1)
            >>> program.add(ops.RX(theta), 0)
            >>> backend = fq.simulator.Simulator("SV")
            >>> results = backend.run_sweep(
            ...     program,
            ...     {theta: [0.0, 0.5]},
            ...     shots=0,
            ...     result_config={"counts": False, "final_state": True},
            ... ).result()
            >>> len(results)
            2
            >>> ["statevector" in result.available_data for result in results]
            [True, True]
        """
        rows = _normalize_parameter_batch(program.operations, bindings)
        results: list[Result] = []
        for row in rows:
            bound = program._assign_normalized_parameters(row)
            point_job = self.run(
                bound,
                shots=shots,
                resource_layout=resource_layout,
                initial_state=initial_state,
                simulation_config=simulation_config,
                result_config=result_config,
            )
            try:
                results.append(point_job.result())
            except BaseException as exc:
                return Job(status="ERROR", error=exc)
        return Job(status="DONE", result=results)

    # --- validation (raises directly from run) ---
    def _validate_initial_state(
        self, initial_state: Any, system_dims: tuple[int, ...]
    ) -> np.ndarray | None:
        """Check an ``initial_state`` against this method, and return it as an array.

        Only the *shape* is checked. Normalization, hermiticity and positivity
        are deliberately not enforced, because nothing downstream relies on
        them: sampling derives its distribution defensively - a statevector run
        divides ``|psi|**2`` by its own sum, and a density-matrix run clips the
        diagonal at zero before doing the same - so an unusual operator evolves
        and samples without special handling. Refusing one would only stop
        somebody running the same arithmetic on a matrix we have no reason to
        object to, and there is no Hermitian-only optimization here that would
        make the refusal honest.

        One consequence worth knowing: given an unnormalized input, an exported
        final state is faithfully unnormalized while counts come from the
        normalized distribution. Both are self-consistent; they answer
        different questions.

        Operator methods are rejected. ``unitary`` and ``superop`` compute the
        program's map rather than a state evolving under it, so there is
        nothing for a starting state to be.
        """
        if initial_state is None:
            return None
        if self._is_operator:
            raise BackendValidationError(
                f"initial_state is not meaningful for method={self._state_field!r}, "
                "which computes the program's map rather than a state evolving "
                "under it. Use method='statevector' or method='density_matrix'"
            )
        state = np.asarray(initial_state, dtype=complex)
        size = 1
        for dimension in system_dims:
            size *= dimension
        # A density matrix also takes a ket, read as the pure state it
        # describes, so one array can start both representations.
        allowed = (
            ((size,), (size, size))
            if self._state_field == "density_matrix"
            else ((size,),)
        )
        if state.shape not in allowed:
            expected = " or ".join(str(shape) for shape in allowed)
            raise BackendValidationError(
                f"initial_state has shape {state.shape}, but this program has "
                f"subsystem dimensions {tuple(system_dims)}, so "
                f"method={self._state_field!r} needs {expected}"
            )
        return state

    def _validate(
        self,
        config: _ResultConfig,
        shots: int,
        facts: _PlanFacts,
        *,
        initial_occupied: frozenset[int] | None,
    ) -> _ResultRequest:
        """Validate result-config / shots constraints against the lowered program.

        Operation support, stochasticity, and semantic execution shape were
        already translated into common facts by the selected backend.
        """
        self._validate_method_support(
            config,
            facts,
            initial_occupied=initial_occupied,
        )
        stochastic = facts.stochastic_final_state
        counts, final_state = _resolve_result_flags(
            config,
            has_measurement=facts.has_measurement,
            stochastic_final_state=stochastic,
        )
        request = self._request_cls(counts=counts, **{self._state_field: final_state})
        requested_state = config.final_state is True

        # shots is only checked when the result actually depends on it: counts
        # always sample per shot, and a stochastic state export needs shots==1
        # below. A non-stochastic state-only request ignores shots entirely
        # (see the engine's per-shot path), so any value - including 0 - is fine.
        common_stochastic = facts.has_measurement or (
            self._nonunitary_is_stochastic and (facts.has_reset or facts.has_channel)
        )
        if stochastic and (not common_stochastic or not facts.has_measurement):
            stochastic_sources = "stochastic execution"
        elif self._nonunitary_is_stochastic:
            stochastic_sources = "measurement, reset, or channel noise"
        else:
            stochastic_sources = "measurement"
        _validate_result_shots(
            counts=request.counts,
            explicit_final_state=requested_state,
            stochastic_final_state=stochastic,
            shots=shots,
            shots_type_error=(
                "shots must be an int when requested results depend on it, "
                f"got {shots!r}"
            ),
            state_label=self._state_field,
            stochastic_sources=stochastic_sources,
        )
        return request

    def _validate_method_support(
        self,
        config: _ResultConfig,
        facts: _PlanFacts,
        *,
        initial_occupied: frozenset[int] | None,
    ) -> None:
        """Reject programs and requests the chosen method cannot represent.

        Raises:
            BackendValidationError: If the lowered program or the result
                request uses something this method cannot execute.
        """
        if not self._is_operator:
            return
        method = self._state_field
        if facts.has_measurement:
            raise BackendValidationError(
                f"method={method!r} cannot execute a measurement; it computes "
                "the program's operator, which no measurement outcome is part "
                "of (use method='statevector' or 'density_matrix' to sample "
                "outcomes)"
            )
        if facts.has_condition:
            raise BackendValidationError(
                f"method={method!r} cannot execute a feedforward condition; it "
                "has no classical register to evaluate one against"
            )
        if config.counts is True:
            raise BackendValidationError(
                f"method={method!r} cannot produce counts; it computes the "
                "program's operator rather than sampling outcomes from it"
            )
        if not self._executes_nonunitary and (facts.has_reset or facts.has_channel):
            source = "reset" if facts.has_reset else "channel noise"
            raise BackendValidationError(
                f"method={method!r} cannot execute {source}; a unitary cannot "
                "represent a non-unitary map (use method='superop' for the "
                "program's channel)"
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
    def _execute_engine(
        self,
        *,
        plan: tuple[ResolvedStep, ...],
        deferred_measurements: tuple[tuple[int, int], ...],
        context: _ExecutionContext,
        policy: _ExecutionPolicy,
    ) -> RawResult:
        """Materialize once in the parent, then dispatch the opaque payload."""
        local_policy = _materialization_policy(policy)
        payload = self._engine.materialize_execution(
            plan,
            system_dims=context.system_dims,
            n_clbits=context.n_clbits,
            deferred_measurements=deferred_measurements,
            policy=local_policy,
        )
        return _dispatch_execution(self._engine, context, payload, policy)

    def _assemble_result(
        self,
        *,
        raw: RawResult,
        config: _ResultConfig,
        simulation: _SimulationConfig,
        lowering: _LoweringContext,
        written_clbits: frozenset[int],
        request: _ResultRequest,
        shots: int,
    ) -> Result:
        """Build one public result without depending on execution routing."""
        engine_allocation = lowering.engine_allocation
        classical_allocation = lowering.classical_allocation
        classical_dims = classical_allocation.classical_dims
        n_clbits = classical_allocation.n_clbits
        counts = None
        state = raw.state
        state_requested = getattr(request, self._state_field)
        available: set[str] = set()
        if request.counts:
            counts = counts_dict_from_arrays(raw.outcome_keys, raw.outcome_counts)
            available.add("counts")
        if state_requested:
            available.add(self._state_field)

        # Counts produced, some clbit never written, no state.
        if request.counts and self._state_field not in available:
            if any(c not in written_clbits for c in range(n_clbits)):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    stacklevel=3,
                )

        extra_data = self._additional_result_data(
            config=config,
            simulation=simulation,
            raw=raw,
        )
        effective_result_config = asdict(config)
        effective_result_config.update(
            counts=request.counts,
            final_state=state_requested,
        )
        metadata = {
            "shots": shots,
            "backend_name": type(self).__name__,
            "method": self._state_field,
            "runtime": self._runtime,
            "simulation_config": asdict(simulation),
            "result_config": effective_result_config,
        }
        if state_requested:
            metadata["state_axes"] = _describe_state_axes(
                engine_allocation,
                lowering.resource_layout,
            )
        return Result(
            counts=counts,
            available=frozenset(available),
            classical_dims=classical_dims,
            data=extra_data,
            metadata=metadata,
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
    ) -> list[ResolvedStep]:
        """Lower a program into an execution plan in one pass.

        Dispatches each instruction to matrix-planning helpers, threading this
        backend's noise model / implementation map / channel map through
        explicitly - none of the three is overridden by any backend today,
        so they take their dependencies as plain parameters instead of
        reading `self`. `Barrier` is recognized by type and skipped
        entirely here - it is a compiler-facing marker with no simulation
        semantics, so it emits no step and cannot affect execution strategy
        or result defaults. Plan facts are derived from the finished plan by
        `_lower_program`, so they cannot drift from what will execute.

        The caller supplies a scalar-only instruction stream and the run's
        private lowering context. `context.resource_layout` is used for
        `MatrixImplementationMap` lookup (`device_operands`) and for
        `NoiseModel._noise_for_occurrence()` physical-selector matching (against the
        occurrence's program target refs); `context.engine_allocation` is
        used for every execution index/dimension - `ApplyMatrixStep`/
        `MeasurementStep`/`ResetStep` targets and conditions. Grouped
        frontend operations are expanded before this method is called.
        """
        resource_layout = context.resource_layout
        engine_allocation = context.engine_allocation
        classical_allocation = context.classical_allocation
        plan: list[ResolvedStep] = []

        for step in operations:
            if isinstance(step, Measurement):
                plan.append(
                    planning._lower_measurement(
                        step,
                        resource_layout,
                        engine_allocation,
                        classical_allocation,
                        self._noise_model,
                    )
                )
            elif isinstance(step, AppliedOperation):
                if isinstance(step.operation, BarrierGate):
                    continue
                if isinstance(step.operation, ResetGate):
                    plan.append(
                        planning._lower_reset(
                            step,
                            resource_layout,
                            engine_allocation,
                            classical_allocation,
                        )
                    )
                else:
                    plan.extend(
                        planning._lower_gate(
                            step,
                            resource_layout,
                            engine_allocation,
                            classical_allocation,
                            self._impl_map,
                            self._noise_model,
                            self._channel_map,
                        )
                    )

        return plan

    def _analyze_lowered_plan(
        self, plan: tuple[ResolvedStep, ...]
    ) -> tuple[_PlanFacts, frozenset[int] | None]:
        """Translate one lowered plan into common semantics and occupancy."""
        return self._analyze_common_plan_facts(plan), None

    def _analyze_common_plan_facts(
        self,
        plan: Sequence[ResolvedStep],
        *,
        claimed_step_types: tuple[type[object], ...] = (),
    ) -> _PlanFacts:
        """Exhaustively derive runtime-independent matrix execution semantics."""
        execution_shape = "operator" if self._is_operator else "single_pass"
        state_nonunitary_uses_trajectories = (
            not self._is_operator and self._nonunitary_is_stochastic
        )
        measured_indices: set[int] = set()
        deferred_measurements: list[tuple[int, int]] = []
        written_clbits: set[int] = set()
        stochastic_final_state = False
        has_measurement = False
        has_reset = False
        has_channel = False
        has_condition = False

        def require_per_shot() -> None:
            nonlocal execution_shape
            if not self._is_operator:
                execution_shape = "per_shot"

        for step in plan:
            if getattr(step, "condition", None) is not None:
                has_condition = True
                require_per_shot()

            if claimed_step_types and isinstance(step, claimed_step_types):
                continue

            if isinstance(step, MeasurementStep):
                has_measurement = True
                stochastic_final_state = True
                if measured_indices.intersection(step.measured_indices):
                    require_per_shot()
                measured_indices.update(step.measured_indices)
                written_clbits.update(step.classical_indices)
                deferred_measurements.extend(
                    zip(step.measured_indices, step.classical_indices)
                )
                continue

            if isinstance(step, ApplyMatrixStep):
                target_indices = step.target_indices
            elif isinstance(step, ResetStep):
                has_reset = True
                target_indices = step.reset_indices
                if state_nonunitary_uses_trajectories:
                    require_per_shot()
                if self._nonunitary_is_stochastic:
                    stochastic_final_state = True
            elif isinstance(step, ApplyChannelStep):
                has_channel = True
                target_indices = step.target_indices
                if state_nonunitary_uses_trajectories:
                    require_per_shot()
                if self._nonunitary_is_stochastic:
                    stochastic_final_state = True
            else:
                raise TypeError(
                    f"unknown resolved execution step {type(step).__name__}"
                )

            if measured_indices.intersection(target_indices):
                require_per_shot()

        if execution_shape != "single_pass":
            deferred_measurements = []

        return _PlanFacts(
            execution_shape=execution_shape,
            deferred_measurements=tuple(deferred_measurements),
            written_clbits=frozenset(written_clbits),
            stochastic_final_state=stochastic_final_state,
            has_measurement=has_measurement,
            has_reset=has_reset,
            has_channel=has_channel,
            has_condition=has_condition,
        )

    def check_noise_support(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report which parts of a noise model this backend can execute.

        A channel descriptor type is supported exactly when the backend's
        channel implementation map has a rule for it - the map's coverage is
        the capability declaration. A descriptor that distinguishes
        probability and rate parameterization (`AmplitudeDamping`,
        `PhaseDamping`, and `Depolarizing`) is
        reported under a mode-qualified label such as ``"AmplitudeDamping(p)"``
        rather than its bare class name: this matrix family never resolves a
        rate without a duration, so a rate-mode instance is unsupported even
        when the class itself has a registered rule. That lets a model
        mixing both modes of the same class appear in both
        `accepted_sources` and `rejected_sources` without ambiguity. Labels
        are display identities only, never registry keys, and are
        deduplicated while preserving first registration order.

        Background registrations are rejected because this backend has no
        continuous-time evolution model. Reset-bound noise cannot pass model
        admission.

        Args:
            noise_model: The noise model to check; it is not executed.

        Returns:
            A frozen report naming accepted and rejected sources.

        Raises:
            BackendValidationError: If ``noise_model`` is not a
                :class:`~fatqat.NoiseModel`.
        """
        if not isinstance(noise_model, NoiseModel):
            raise BackendValidationError("noise_model must be a NoiseModel")
        accepted: list[str] = []
        rejected: list[str] = []
        warnings_: list[str] = []
        seen: set[str] = set()

        def _record(label: str, supported: bool, warning: str) -> None:
            if label in seen:
                return
            seen.add(label)
            (accepted if supported else rejected).append(label)
            if not supported:
                warnings_.append(warning)

        for channel, operation in noise_model._noise_sources():
            channel_type = type(channel)
            background = operation is None
            built_in_damping = channel_type in (
                AmplitudeDamping,
                PhaseDamping,
                Depolarizing,
            )
            rate_mode = built_in_damping and channel.rate is not None
            qualifiers: list[str] = []
            if built_in_damping:
                qualifiers.append("rate" if rate_mode else "p")
            if background:
                qualifiers.append("background")
            label = channel_type.__name__
            if qualifiers:
                label += f"({', '.join(qualifiers)})"
            if isinstance(channel, Loss):
                if self._supports_loss:
                    _record(label, True, "")
                else:
                    _record(
                        label,
                        False,
                        f"{label} is not supported: this backend does not model "
                        "carrier loss (use AtomArraySimulator)",
                    )
            elif background:
                _record(
                    label,
                    False,
                    f"{label} is not supported: this matrix backend has no "
                    "continuous-time evolution model",
                )
            elif channel_type is ThermalRelaxation:
                _record(
                    label,
                    False,
                    f"{label} is rejected by matrix-family policy because it "
                    "is a generator/time declaration; a registered channel "
                    "implementation does not override that policy. Explicitly "
                    "convert it with as_channels(duration)",
                )
            elif self._channel_map.get(channel_type) is None:
                _record(
                    label,
                    False,
                    f"{label} has no channel implementation on this backend",
                )
            elif rate_mode:
                _record(
                    label,
                    False,
                    f"{label} is not supported: rate mode has no matrix-backend "
                    "Kraus implementation on this backend",
                )
            else:
                _record(label, True, "")
        if noise_model._readout_confusions():
            accepted.append("ReadoutConfusion")
        return NoiseSupportReport(
            supported=not rejected,
            accepted_sources=tuple(accepted),
            rejected_sources=tuple(rejected),
            warnings=tuple(warnings_),
        )
