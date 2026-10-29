"""Matrix-based gate-level simulation.

``Simulator`` runs gate-level programs with a NumPy or Numba runtime. For
pulse-resolved physical simulation, use the models in ``fatqat.emulator``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import batched, chain
from typing import Any

import numpy as np

from .._expectation import (
    _ExpectationExecution,
    _TermOccurrence,
    _combine_term_statistics,
    _plan_term_occurrences,
    _reduce_outcome_counts,
)
from .._parameter_binding import (
    _normalize_parameter_batch,
    _raise_for_unbound_parameters,
)
from ..errors import BackendValidationError, UnsupportedOperationError
from ..execution import ExecutableProgram
from .._index_allocation import (
    _ClassicalAllocation,
    _EngineAllocation,
    _describe_state_axes,
)
from ..implementation import MatrixImplementationMap, default_matrix_implementation_map
from ..job import Job
from ..noise import (
    AmplitudeDamping,
    Depolarizing,
    PhaseDamping,
    TransitionRelaxation,
    NoiseModel,
    Loss,
    ThermalRelaxation,
)
from ..observable import Observable
from ..operations import Barrier, CustomOperation, Measurement, Reset
from ..parameters import Parameter, ParameterVector
from ..program import Program, _AppliedOperation
from ..registers import RegisterRef
from ..resource_layout import DeviceOperand, ResourceLayout
from ..result import (
    Result,
    counts_dict_from_arrays,
)
from ._engine.engines.base import MatrixEngine
from .._backends.backend_utils import (
    _LoweringContext,
    _canonicalize_method,
    _normalize_config,
    _resolve_result_flags,
    _validate_result_shots,
)
from . import planning
from ._engine.state import EvolutionState
from ._execution_contract import (
    _RunContext,
    _PlanFacts,
    RawResult,
    _ResultRequest,
    _SimulationConfig,
    _SimulatorResultConfig,
)
from .._backends.view_normalization import ProgramInstruction, _break_grouped_operations
from .._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    CustomOperationStep,
    MeasurementStep,
    ResetStep,
    ResolvedStep,
)


@dataclass(frozen=True)
class _MethodSpec:
    """Everything the chosen simulation method binds into a `Simulator`.

    Attributes:
        numpy_engine_name: Class name in the lazily loaded NumPy engine module.
        numba_engine_name: Class name in the lazily loaded Numba engine module.
    """

    numpy_engine_name: str
    numba_engine_name: str


_METHOD_SPECS: dict[str, _MethodSpec] = {
    "statevector": _MethodSpec(
        numpy_engine_name="NumpySVEngine",
        numba_engine_name="NumbaSVEngine",
    ),
    "density_matrix": _MethodSpec(
        numpy_engine_name="NumpyDMEngine",
        numba_engine_name="NumbaDMEngine",
    ),
    "unitary": _MethodSpec(
        numpy_engine_name="NumpyUnitaryEngine",
        numba_engine_name="NumbaUnitaryEngine",
    ),
    "superop": _MethodSpec(
        numpy_engine_name="NumpySuperopEngine",
        numba_engine_name="NumbaSuperopEngine",
    ),
}

# Base states a sampled expectation sweep keeps alive while its terms sample.
_SAMPLED_BASE_STATE_BYTES = 256 * 2**20


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    """Program execution facts shared by every Simulator consumer."""

    plan: tuple[ResolvedStep, ...]
    facts: _PlanFacts
    lowering: _LoweringContext
    simulation: _SimulationConfig
    initial_state: EvolutionState[Any]


@dataclass(frozen=True, slots=True)
class _BoundExpectationOccurrence:
    """One logical term occurrence bound to matrix-engine resources."""

    occurrence: _TermOccurrence
    engine_factors: tuple[tuple[int, str], ...]
    scratch_indices: tuple[int, ...]
    tail: tuple[ResolvedStep, ...]


class Simulator:
    """Run a gate-level program with matrices and finite channels.

    ``statevector`` (``SV``) and ``density_matrix`` (``DM``) evolve a state.
    Statevector simulation samples reset and channel branches; density-matrix
    simulation applies them exactly. ``unitary`` and ``superop`` compute the
    program's map instead. Operator methods reject measurement, feedforward,
    counts, and an initial state; ``unitary`` also rejects reset and channels.

    A ``superop`` result uses column-stacking vectorization, so apply it to
    ``rho.reshape(-1, order="F")``.

    The generic simulator accepts qubits, qudits, and mixed local dimensions.
    Hardware-profile subclasses narrow the native gate set and resource layout
    while preserving the same run and result contract.

    A backend may be reused for sequential runs. Use a separate instance for
    each concurrent caller.

    Examples:
        Run a one-qubit program with density-matrix simulation:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(1)
        >>> program.add(ops.H, 0)
        >>> result = fq.simulator.Simulator(method="DM").run(program).result()
        >>> result.get_density_matrix()
        array([[0.5+0.j, 0.5+0.j],
               [0.5+0.j, 0.5+0.j]])
    """

    # Subclasses may replace this with a dataclass derived from
    # ``_SimulatorResultConfig`` to expose hardware-specific result artifacts.
    # The normalizer derives accepted keys from this schema, so unsupported
    # fields fail before execution.
    _result_config_cls: type[_SimulatorResultConfig] = _SimulatorResultConfig

    # Subclasses may replace this with a dataclass derived from
    # ``_SimulationConfig`` for hardware-specific simulator controls. The
    # base backend consumes the inherited matrix-execution fields.
    _simulation_config_cls: type[_SimulationConfig] = _SimulationConfig

    # Whether this backend implements the per-shot atom-occupancy lifecycle
    # Loss needs (loading, per-shot loss, refill). False on the generic
    # matrix backends, which reject Loss via validate_noise_model rather than
    # silently ignoring it; AtomArraySimulator sets it True.
    _supports_loss: bool = False

    def __init__(
        self,
        method: str = "statevector",
        *,
        runtime: str = "numba",
        implementation_map: MatrixImplementationMap | None = None,
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a gate-level simulator.

        Args:
            method: ``"statevector"``, ``"density_matrix"``, ``"unitary"``, or
                ``"superop"``. Names are case-insensitive; ``"SV"`` and
                ``"DM"`` are accepted aliases.
            runtime: Numerical engine, case-insensitive. ``"numba"`` (the
                default) lazily JIT-compiles its kernels and provides the
                runtime's threaded and fusion paths. ``"numpy"`` executes the
                reference kernels directly without JIT warm-up. Both support
                every method, but are not promised to be bit-identical.
            implementation_map: Matrix, channel, and custom-operation implementations.
                ``None`` uses ``default_matrix_implementation_map()``.
            noise: Noise model applied to every run. ``None`` means ideal
                execution. The model is copied at construction: ``add()``
                calls made on the original after the backend is built do not
                reach this backend - build the backend after the model is
                complete, or build a new backend.

        Raises:
            BackendValidationError: If ``method`` or ``runtime`` is not one
                of the supported names, the required numba dependency cannot
                be imported, or the captured noise model contains a source
                this backend cannot execute.
        """
        normalized = _canonicalize_method(method, _METHOD_SPECS)
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
        # Select implementations here; execution support belongs to the engine.
        spec = _METHOD_SPECS[normalized]
        if normalized_runtime == "numba":
            try:
                from ._engine.engines import numba_engine as engine_module
            except ImportError as exc:
                raise BackendValidationError(
                    "runtime='numba' requires the numba dependency; reinstall "
                    "fatqat to repair the environment"
                ) from exc
            engine_name = spec.numba_engine_name
        else:
            from ._engine.engines import numpy_engine as engine_module

            engine_name = spec.numpy_engine_name
        self._engine_cls: type[MatrixEngine] = getattr(engine_module, engine_name)
        self._runtime = normalized_runtime

        if implementation_map is None:
            implementation_map = default_matrix_implementation_map()
        self._impl_map = implementation_map.copy()
        if noise is not None and not isinstance(noise, NoiseModel):
            raise BackendValidationError("noise must be a NoiseModel or None")
        source_noise = noise if noise is not None else NoiseModel()
        self._noise_model = source_noise._copy()
        self.validate_noise_model(self._noise_model)
        # The engine object and its dimension-dependent numeric caches are
        # reused, while every execution allocates or resets evolving state.
        # A backend instance is not safe for concurrent run() calls.
        self._engine = self._engine_cls()
        self._state_field = self._engine.capabilities.quantum.representation

    @property
    def method(self) -> str:
        """Canonical simulation method selected at construction.

        The value is ``"statevector"``, ``"density_matrix"``, ``"unitary"``,
        or ``"superop"`` regardless of the alias originally supplied. It also
        appears in result metadata and names the method-native result field.

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
        # NumPy/Numba keep private little-endian prefix strides. Reversing the
        # public (operand, dimension) pairs together makes those prefixes equal
        # public MSB suffix strides, so native results need no ordering copy.
        modeled = tuple(reversed(modeled))
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
        return self._prepare_program(program, context=context)

    def _prepare_program(
        self,
        program: Program,
        *,
        context: _LoweringContext | None = None,
        parameter_values: Mapping[Parameter, np.ndarray] | None = None,
    ) -> tuple[tuple[ResolvedStep, ...], _PlanFacts]:
        """Lower and freeze once, then derive common plan facts.

        With ``parameter_values`` (a sweep), parameter-holding gates lower to
        `ApplyParametricMatrixStep` values carrying their value columns, which
        the engine binds per row. Plan facts read only step kinds, indices and
        conditions, so they are valid for every row. Without it, the program
        must be fully bound.
        """
        if context is None:
            resource_layout = self._resolve_resource_layout(program)
            context = _LoweringContext(
                resource_layout=resource_layout,
                engine_allocation=self._allocate_engine_indices(
                    program, resource_layout
                ),
                classical_allocation=_ClassicalAllocation.from_program(program),
            )
        operations = _break_grouped_operations(program._instructions)
        if parameter_values is None:
            _raise_for_unbound_parameters(program._instructions)
        # pylint: disable-next=fixme
        # TODO: Support custom operations in run_sweep.
        elif any(
            isinstance(step, _AppliedOperation)
            and isinstance(step.operation, CustomOperation)
            for step in operations
        ):
            raise UnsupportedOperationError(
                "run_sweep does not support custom operations"
            )
        plan = tuple(self._lower(operations, context))
        if parameter_values is not None:
            plan = planning._inject_parameter_values(plan, parameter_values)
        return plan, self._analyze_lowered_plan(plan)

    def run(
        self,
        program: Program | ExecutableProgram,
        *,
        shots: int = 1024,
        resource_layout: ResourceLayout | None = None,
        initial_state: Any = None,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job[Result]:
        """Validate and execute one program.

        The returned job is eager and already terminal. Program and option
        validation errors raise directly. Execution errors follow the
        ``error_handling`` option described below.

        Args:
            program: Program to execute, or an executable compiler result that
                carries its required resource layout.
            shots: Circuit repetitions. Counts require a positive integer. An
                explicitly requested stochastic final state requires exactly
                one shot; deterministic state-only runs ignore this value.
            resource_layout: Assignment from every program quantum reference
                to a device label. It must be complete, one-to-one, and
                compatible with the backend. The documented backend default
                is used when omitted.
            initial_state: Quantum array or ``EvolutionState`` supplying quantum
                and classical data for every shot. None selects the ground state;
                see ``EvolutionState`` for classical defaults and shot independence.
                Statevector execution accepts shape ``(D,)``; density-matrix
                execution accepts ``(D, D)`` or a ``(D,)`` ket. Arrays use public
                most-significant-first basis order. Only the quantum shape is
                validated, not normalization or physical validity. Operator
                methods reject an initial state.
                Shape errors raise directly. Runtime array-conversion failures
                follow ``error_handling``.
            simulation_config: Optional per-run execution controls. String
                choices are case-sensitive. Accepted keys are:

                - ``"seed"`` (``int | None``, default ``None``): Use a
                  non-negative sampling seed. Booleans are rejected; ``None``
                  uses fresh entropy. A negative value is rejected during
                  execution and follows ``error_handling``. A
                  fixed seed reproduces counts only for the same method,
                  runtime, shot count, parallelism settings, and result
                  requests.
                - ``"shot_parallelism"`` (``str``, default ``"auto"``): How
                  independent shots run: ``"auto"``, ``"serial"``,
                  ``"threads"``, or ``"processes"``. Explicit parallel modes
                  require an eligible counts-only run with at least two shots
                  and workers. Threads require a compatible Numba statevector
                  run.
                - ``"kernel_parallelism"`` (``str``, default ``"auto"``):
                  Numerical work within one shot: ``"auto"``, ``"serial"``,
                  or ``"threads"``. Threads require Numba and cannot be
                  requested with parallel shots.
                - ``"max_workers"`` (``int | None``, default ``None``):
                  Positive concurrency limit. ``None`` uses the runtime
                  limit; ``1`` conflicts with an explicit parallel request.
                - ``"error_handling"`` (``str``, default ``"raise"``):
                  ``"raise"`` propagates execution and result-assembly errors
                  immediately. ``"capture"`` returns a failed job whose
                  ``result()`` re-raises the original exception. Input and
                  configuration validation always raise directly. Engine
                  planning errors follow this option.
                - ``"fusion"`` (``bool``, default ``False``): Whether to
                  combine compatible adjacent operations. ``True`` requires
                  Numba with ``density_matrix``, ``unitary``, or ``superop``.

                Unknown or incompatible entries are rejected.
            result_config: Optional output requests. Accepted keys are:

                - ``"counts"`` (``bool | None``, default ``None``): Histogram
                  of the final classical register. ``True`` requests it,
                  ``False`` suppresses it, and ``None`` enables it when the
                  program measures. Counts require an integer ``shots > 0``.
                - ``"final_state"`` (``bool | None``, default ``None``):
                  Method-native state or map. ``True`` requests it, ``False``
                  suppresses it, and ``None`` enables it when that artifact is
                  deterministic. A requested stochastic state requires
                  ``shots == 1``.
                - ``"final_classical_state"`` (``bool``, default ``False``):
                  Whether to return one shot's final ``ClassicalState``
                  (register digits and site occupancy). It is stochastic under
                  the same conditions as the final state and then requires
                  ``shots == 1``. Operator methods reject it.

        Returns:
            A completed ``Job`` whose result is a ``Result``.

        Raises:
            BackendValidationError: If the program, resource layout, noise
                selectors, or requested configuration is invalid.
            MatrixImplementationError: If a matrix rule cannot build the
                selected operation for its target dimensions.
            TypeError: If ``program`` is not a ``Program`` or
                ``ExecutableProgram``, or if ``simulation_config`` or
                ``result_config`` is not a dictionary or ``None``.
            UnsupportedOperationError: If the program contains an operation
                without a backend implementation, or one whose target key is
                illegal for this backend.
            ValueError: If ``resource_layout`` is supplied for an executable
                compiler result that already carries one.
        """
        if isinstance(program, ExecutableProgram):
            if resource_layout is not None:
                raise ValueError(
                    "resource_layout cannot be supplied with an ExecutableProgram"
                )
            resource_layout = program.resource_layout
            program = program.program
        elif not isinstance(program, Program):
            raise TypeError("program must be a Program or ExecutableProgram")

        _raise_for_unbound_parameters(program._instructions)
        execution = self._prepare_execution(
            program,
            resource_layout=resource_layout,
            initial_state=initial_state,
            simulation_config=simulation_config,
        )
        config, context = self._prepare_run(
            execution, shots=shots, result_config=result_config
        )
        job = self._execute_plan(execution, config, context)
        if job.status != "DONE":
            return job
        (result,) = job.result()
        return Job(status="DONE", result=result)

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
        """Bind and run every row of a complete parameter batch.

        Single parameters accept shape ``(N,)`` and parameter vectors of
        length ``M`` accept shape ``(N, M)``. All entries must use the same
        ``N``. Rows run in input order with the same options. The batch is
        validated before execution and stored as NumPy arrays, so a custom
        matrix rule may receive a row's value as a NumPy scalar. One explicit
        seed is reused for every row, so sampled row errors can be correlated.
        Batch validation errors raise directly. Failures while binding or
        executing any row follow ``run()``'s error-handling policy; no partial
        list is exposed.

        Args:
            program: Parameterized template program.
            bindings: Complete mapping from each ``Parameter`` or
                ``ParameterVector`` in the program to its batch values.
            shots: Number of repetitions forwarded to every row.
            resource_layout: Complete, one-to-one assignment of program
                quantum references to compatible device labels, reused for
                every row. The backend default is used when omitted.
            initial_state: Quantum array or ``EvolutionState`` used to initialize
                every shot in every row, following the same rules as ``run()``.
            simulation_config: Controls reused for every row. Accepted keys:

                - ``"seed"`` (``int | None``, default ``None``): Use a
                  non-negative sampling seed; booleans are rejected. ``None``
                  uses fresh entropy for each row. A negative value fails
                  during execution and follows ``error_handling``.
                - ``"shot_parallelism"`` (``str``, default ``"auto"``):
                  ``"auto"``, ``"serial"``, ``"threads"``, or
                  ``"processes"``. Explicit parallel modes require an
                  eligible counts-only run with at least two shots and
                  workers. Threads require a compatible Numba statevector run.
                - ``"kernel_parallelism"`` (``str``, default ``"auto"``):
                  ``"auto"``, ``"serial"``, or ``"threads"``. Threads
                  require Numba and cannot be requested with parallel shots.
                - ``"max_workers"`` (``int | None``, default ``None``):
                  Positive concurrency limit. ``None`` uses the runtime
                  limit; ``1`` conflicts with an explicit parallel request.
                - ``"error_handling"``: Same values, default, and behavior as
                  ``run()``.
                - ``"fusion"`` (``bool``, default ``False``): Whether to
                  combine compatible operations. ``True`` requires Numba with
                  ``density_matrix``, ``unitary``, or ``superop``.

                Unknown or incompatible entries are rejected.
            result_config: Output requests reused for every row. Accepted keys:

                - ``"counts"`` (``bool | None``, default ``None``): Histogram
                  of the final classical register. ``None`` enables it when
                  the program measures; counts require an integer
                  ``shots > 0``.
                - ``"final_state"`` (``bool | None``, default ``None``):
                  Method-native state or map. ``None`` enables it when the
                  result is deterministic; a requested stochastic state
                  requires ``shots == 1``.
                - ``"final_classical_state"`` (``bool``, default ``False``):
                  Each row's final ``ClassicalState``, with the same rules as
                  ``run()``.

                ``True`` requests a field and ``False`` suppresses it.

        Returns:
            An eager job carrying an ordered list of row results.

        Raises:
            TypeError: If ``bindings`` is not an object-keyed mapping, a batch
                contains values other than built-in ``int``/``float`` or NumPy
                integer/floating scalars, or either configuration is not a
                dictionary or ``None``.
            ValueError: If the program is not parameterized, assignments are
                missing or duplicated, or batch ranks and lengths disagree.
            BackendValidationError: If a bound row or forwarded run option
                fails normal Simulator validation.
            MatrixImplementationError: If a matrix rule cannot build an
                operation for one row and ``error_handling`` is ``"raise"``.
            UnsupportedOperationError: If the program contains custom operations
                or a row contains an operation the backend cannot run.

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
        columns = _normalize_parameter_batch(program._instructions, bindings)
        # Structural resolution and lowering happen once for the whole batch,
        # on the same direct-raise validation path as run(): layout,
        # allocation, noise-selector validation, and rule selection read
        # operation types and targets only, so they cannot depend on θ values.
        execution = self._prepare_execution(
            program,
            resource_layout=resource_layout,
            initial_state=initial_state,
            simulation_config=simulation_config,
            parameter_values=columns,
        )
        config, context = self._prepare_run(
            execution, shots=shots, result_config=result_config
        )
        return self._execute_plan(execution, config, context)

    def _prepare_run(
        self,
        execution: _PreparedExecution,
        *,
        shots: int,
        result_config: dict[str, Any] | None,
    ) -> tuple[_SimulatorResultConfig, _RunContext]:
        """Resolve direct results and shot requirements before execution.

        Called once per run or sweep batch on the direct-raise validation path.
        The returned settings do not depend on parameter values; sweeps reuse
        them while the engine binds and executes each row.
        """
        config = _normalize_config(
            result_config,
            self._result_config_cls,
            "result_config",
            backend_name=type(self).__name__,
        )
        request = self._validate_run_request(config, shots, execution.facts)
        context = _RunContext(
            request=request,
            system_dims=tuple(execution.lowering.engine_allocation.system_dims),
            n_clbits=execution.lowering.classical_allocation.n_clbits,
            shots=shots,
            seed=execution.simulation.seed,
        )
        return config, context

    def _prepare_execution(
        self,
        program: Program,
        *,
        resource_layout: ResourceLayout | None,
        initial_state: Any,
        simulation_config: dict[str, Any] | None,
        parameter_values: Mapping[Parameter, np.ndarray] | None = None,
    ) -> _PreparedExecution:
        """Resolve and lower one program for direct or derived execution."""
        simulation = _normalize_config(
            simulation_config,
            self._simulation_config_cls,
            "simulation_config",
            backend_name=type(self).__name__,
        )
        # Both hooks are resolved exactly once per run, before any execution
        # try block: capacity, dimension, grid-fit, and mapping failures must
        # raise directly, never become a failed Job. The resource layout is the
        # public-facing effective mapping (available to backend validation);
        # the engine index allocation stays private to execution preparation.
        # Both are paired into one private lowering context and threaded
        # through preparation/lowering unchanged, so lowering never re-resolves
        # either value.
        resource_layout = self._resolve_resource_layout(program, resource_layout)
        engine_allocation = self._allocate_engine_indices(program, resource_layout)
        classical_allocation = _ClassicalAllocation.from_program(program)
        initial_state = self._prepare_initial_state(initial_state)
        self._validate_initial_state(
            initial_state,
            tuple(reversed(engine_allocation.system_dims)),
            classical_allocation.n_clbits,
        )
        # Selector-identity validation runs immediately after the effective
        # resource layout is known and before any lowering/plan step is built.
        # A foreign ref or a label outside the backend's device universe fails
        # on this direct-raise path; a legal label unused by this layout remains
        # a valid no-op.
        self._noise_model._validate_for(
            program, self._legal_device_operands(program, resource_layout)
        )
        lowering = _LoweringContext(
            resource_layout=resource_layout,
            engine_allocation=engine_allocation,
            classical_allocation=classical_allocation,
        )
        plan, facts = self._prepare_program(
            program, context=lowering, parameter_values=parameter_values
        )
        prepared = _PreparedExecution(
            plan=plan,
            facts=facts,
            lowering=lowering,
            simulation=simulation,
            initial_state=initial_state,
        )
        self._validate_engine_support(prepared)
        return prepared

    def _prepare_initial_state(self, initial: Any) -> EvolutionState[Any]:
        """Snapshot public evolution data; preserve quantum-only input support."""
        # pylint: disable-next=fixme
        # TODO: clean this up, being an operator engine does not inherently rule out custom behavior.
        if initial is not None and self._engine.capabilities.quantum.is_operator:
            raise BackendValidationError(
                f"initial_state is not meaningful for method={self._state_field!r}, "
                "which computes the program's map rather than a state evolving "
                "under it. Use method='statevector' or method='density_matrix'"
            )
        if isinstance(initial, EvolutionState):
            return EvolutionState(
                quantum=initial.quantum, classical=deepcopy(initial.classical)
            )
        return EvolutionState(quantum=initial)

    def _run_expectation(
        self,
        program: Program,
        observables: tuple[Observable, ...],
        *,
        shots: int,
        simulation_config: dict[str, Any] | None,
        parameter_values: Mapping[Parameter, np.ndarray] | None = None,
    ) -> Job[list[_ExpectationExecution]]:
        """Execute a private exact or sampled expectation request per entry.

        ``parameter_values`` are a sweep's validated columns; without them the
        program is one fully bound entry. The program is prepared once and
        every entry evolves in one engine call; sampled values then run each
        term's measurement tail from every entry's state.
        """
        execution = self._prepare_execution(
            program,
            resource_layout=None,
            initial_state=None,
            simulation_config=simulation_config,
            parameter_values=parameter_values,
        )
        # pylint: disable-next=fixme
        # TODO: Support custom operations
        if any(isinstance(step, CustomOperationStep) for step in execution.plan):
            raise UnsupportedOperationError(
                "expectation values are not supported for custom operations"
            )
        constants, bound_occurrences = self._prepare_expectation(
            execution, program, observables, shots=shots
        )
        try:
            if shots == 0 or not bound_occurrences:
                entries = [
                    (values, (0.0,) * len(values))
                    for values in self._execute_exact_expectation(
                        execution,
                        constants,
                        bound_occurrences,
                        shots=shots,
                        n_entries=(
                            len(next(iter(parameter_values.values())))
                            if parameter_values
                            else 1
                        ),
                    )
                ]
            else:
                entries = self._execute_sampled_expectation(
                    execution, constants, bound_occurrences, shots=shots
                )
            return Job(
                status="DONE",
                result=[
                    _ExpectationExecution(
                        values,
                        standard_errors,
                        method=self._state_field,
                        runtime=self._runtime,
                    )
                    for values, standard_errors in entries
                ],
            )
        except Exception as exc:
            if execution.simulation.error_handling == "raise":
                raise
            return Job(status="ERROR", error=exc)

    def _prepare_expectation(
        self,
        execution: _PreparedExecution,
        program: Program,
        observables: tuple[Observable, ...],
        *,
        shots: int,
    ) -> tuple[tuple[float, ...], tuple[_BoundExpectationOccurrence, ...]]:
        """Bind observables and prepare expectation work from a lowered program."""
        if self._engine.capabilities.quantum.is_operator:
            raise UnsupportedOperationError(
                f"method={self._state_field!r} computes an operator, not a state "
                "whose expectation can be evaluated"
            )
        if (
            shots == 0
            and self._state_field == "statevector"
            and execution.facts.stochastic_final_state
        ):
            raise UnsupportedOperationError(
                "an exact statevector expectation is unavailable for stochastic "
                "reset or channel execution; use density_matrix or positive shots"
            )
        if shots > 0 and not self._engine.capabilities.supports_classical_register:
            raise UnsupportedOperationError(
                "sampled expectations require an engine with a classical register"
            )
        occurrences, constants = _plan_term_occurrences(observables)
        bound_occurrences = self._bind_expectation_occurrences(
            program,
            execution,
            occurrences,
            sampled=shots > 0,
        )
        return constants, bound_occurrences

    def _bind_expectation_occurrences(
        self,
        program: Program,
        execution: _PreparedExecution,
        occurrences: tuple[_TermOccurrence, ...],
        *,
        sampled: bool,
    ) -> tuple[_BoundExpectationOccurrence, ...]:
        """Bind logical observable factors to engine and scratch indices."""
        lowering = execution.lowering
        logical_refs = tuple(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        scratch_start = lowering.classical_allocation.n_clbits
        bound_occurrences = []
        for occurrence in occurrences:
            factor_refs = tuple(
                logical_refs[index] for index, _letter in occurrence.logical_factors
            )
            engine_factors = tuple(
                (lowering.engine_index(ref), letter)
                for ref, (_index, letter) in zip(
                    factor_refs, occurrence.logical_factors, strict=True
                )
            )
            if sampled:
                tail = planning._build_expectation_tail(
                    engine_factors,
                    factor_refs,
                    scratch_start=scratch_start,
                    resource_layout=lowering.resource_layout,
                    noise_model=self._noise_model,
                )
            else:
                if (
                    planning._expectation_confusions(
                        factor_refs,
                        lowering.resource_layout,
                        self._noise_model,
                    )
                    is not None
                ):
                    raise UnsupportedOperationError(
                        "an exact expectation cannot apply selected readout "
                        "confusion; use positive shots"
                    )
                tail = ()
            bound_occurrences.append(
                _BoundExpectationOccurrence(
                    occurrence=occurrence,
                    engine_factors=engine_factors,
                    scratch_indices=tuple(
                        range(scratch_start, scratch_start + len(engine_factors))
                    ),
                    tail=tail,
                )
            )
        return tuple(bound_occurrences)

    def _execute_expectation_base(
        self,
        execution: _PreparedExecution,
    ) -> Iterator[EvolutionState]:
        """Evolve every entry once and yield its quantum and classical state."""
        context = _RunContext(
            request=_ResultRequest(
                counts=False, final_state=True, final_classical_state=True
            ),
            system_dims=tuple(execution.lowering.engine_allocation.system_dims),
            n_clbits=execution.lowering.classical_allocation.n_clbits,
            shots=1,
            seed=execution.simulation.seed,
        )
        for raw in self._engine.execute(
            execution.plan,
            context=context,
            simulation_config=execution.simulation,
            initial_state=execution.initial_state,
        ):
            if raw.state is None or raw.classical_state is None:
                raise RuntimeError("expectation base execution returned no state")
            yield EvolutionState(quantum=raw.state, classical=raw.classical_state)

    def _execute_exact_expectation(
        self,
        execution: _PreparedExecution,
        constants: tuple[float, ...],
        bound_occurrences: tuple[_BoundExpectationOccurrence, ...],
        *,
        shots: int,
        n_entries: int,
    ) -> list[tuple[float, ...]]:
        """Contract every observable against each entry's backend-evolved state."""
        plan = execution.plan
        terms_by_observable = [[] for _constant in constants]
        for bound in bound_occurrences:
            terms_by_observable[bound.occurrence.observable_index].append(
                (bound.occurrence.coefficient, bound.engine_factors)
            )
        has_terms = bool(bound_occurrences)
        context = _RunContext(
            request=_ResultRequest(counts=False, final_state=has_terms),
            system_dims=tuple(execution.lowering.engine_allocation.system_dims),
            n_clbits=execution.lowering.classical_allocation.n_clbits,
            shots=1 if has_terms else shots,
            seed=execution.simulation.seed,
        )
        values = self._engine.expectation_values(
            plan,
            tuple(tuple(terms) for terms in terms_by_observable) if has_terms else (),
            context=context,
            simulation_config=execution.simulation,
            initial_state=execution.initial_state,
        )
        if not has_terms:
            return [constants] * n_entries
        return [
            tuple(
                constant + value
                for constant, value in zip(constants, entry_values, strict=True)
            )
            for entry_values in values
        ]

    def _execute_sampled_expectation(
        self,
        execution: _PreparedExecution,
        constants: tuple[float, ...],
        bound_occurrences: tuple[_BoundExpectationOccurrence, ...],
        *,
        shots: int,
    ) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        """Sample each term occurrence for every entry; one result per entry.

        A deterministic plan evolves every entry once and runs each term's
        measurement tail from those states, a bounded chunk of entries at a
        time. A stochastic plan reruns the whole plan with each tail. Each
        term has one seed, shared by entries.
        """

        def with_scratch(
            initial: EvolutionState, bound: _BoundExpectationOccurrence
        ) -> EvolutionState:
            # The tail measures into scratch clbits after the program's own.
            clbits = initial.classical.clbits
            if clbits is None:
                return initial
            return replace(
                initial,
                classical=replace(
                    initial.classical,
                    clbits=clbits + [0] * len(bound.engine_factors),
                ),
            )

        child_seeds = np.random.SeedSequence(execution.simulation.seed).spawn(
            len(bound_occurrences)
        )
        contexts = [
            _RunContext(
                request=_ResultRequest(counts=True, final_state=False),
                system_dims=tuple(execution.lowering.engine_allocation.system_dims),
                n_clbits=(
                    execution.lowering.classical_allocation.n_clbits
                    + len(bound.engine_factors)
                ),
                shots=shots,
                seed=int(child_seed.generate_state(1, dtype=np.uint64)[0]),
            )
            for child_seed, bound in zip(child_seeds, bound_occurrences, strict=True)
        ]

        def sample_terms(
            base_states: Sequence[EvolutionState] | None,
        ) -> list[list[tuple[int, int]]]:
            """Return per-entry term statistics, from base states if given."""
            statistics: list[list[tuple[int, int]]] = []
            for context, bound in zip(contexts, bound_occurrences, strict=True):
                if base_states is not None:
                    raws = self._engine.continue_from_states(
                        bound.tail,
                        [with_scratch(state, bound) for state in base_states],
                        context=context,
                        simulation_config=execution.simulation,
                    )
                else:
                    raws = list(
                        self._engine.execute(
                            execution.plan + bound.tail,
                            context=context,
                            simulation_config=execution.simulation,
                            initial_state=with_scratch(execution.initial_state, bound),
                        )
                    )
                if not statistics:
                    statistics = [[] for _raw in raws]
                for entry, raw in zip(statistics, raws, strict=True):
                    if raw.outcome_keys is None or raw.outcome_counts is None:
                        raise RuntimeError(
                            "sampled expectation execution returned no outcomes"
                        )
                    measured_rows = raw.outcome_keys[:, bound.scratch_indices]
                    entry.append(
                        _reduce_outcome_counts(
                            bound.engine_factors,
                            zip(measured_rows, raw.outcome_counts, strict=True),
                        )
                    )
            return statistics

        if execution.facts.stochastic_final_state:
            # Noise, mid-circuit measurement, or reset gives each shot its own
            # final state, so no evolved state can be shared by the shots.
            # Every term reruns the whole plan with its tail instead.
            statistics = sample_terms(None)
        else:
            # Every shot of a row ends in the same state (density-matrix noise
            # included), so each row evolves once and every term's tail
            # continues from it. Keep a bounded number of base states alive:
            # each chunk is evolved, sampled by every term, then released.
            # Small states fit in one chunk, so their terms run back to back.
            states = self._execute_expectation_base(execution)
            first = next(states)
            rows = max(1, _SAMPLED_BASE_STATE_BYTES // first.quantum.nbytes)
            statistics = []
            for chunk in batched(chain([first], states), rows):
                statistics.extend(sample_terms(chunk))
        occurrences = tuple(bound.occurrence for bound in bound_occurrences)
        return [
            _combine_term_statistics(constants, occurrences, entry, shots=shots)
            for entry in statistics
        ]

    def _execute_plan(
        self,
        execution: _PreparedExecution,
        config: _SimulatorResultConfig,
        context: _RunContext,
    ) -> Job[list[Result]]:
        """Execute a prepared plan for every entry of its parameter batch.

        An ordinary run is a batch of one; a sweep has one entry per set of
        parameter values. Error handling controls whether binding, engine planning,
        execution, and result-assembly failures propagate or become a failed
        job; no partial list is exposed.
        """
        try:
            # Run every entry before assembling any result. Every result is
            # returned, so consuming `execute` lazily saves no memory here, and
            # interleaving assembly with execution measured ~5-10% slower
            # for 6 qubit-circuit.
            raws = list(
                self._engine.execute(
                    execution.plan,
                    context=context,
                    simulation_config=execution.simulation,
                    initial_state=execution.initial_state,
                )
            )
            results = [
                self._assemble_result(
                    raw=raw,
                    config=config,
                    simulation=execution.simulation,
                    lowering=execution.lowering,
                    request=context.request,
                    shots=context.shots,
                )
                for raw in raws
            ]
        except Exception as exc:  # execution-stage failure
            if execution.simulation.error_handling == "raise":
                raise
            return Job(status="ERROR", error=exc)
        return Job(status="DONE", result=results)

    # --- validation (raises directly from run) ---
    def _validate_initial_state(
        self,
        initial_state: EvolutionState,
        system_dims: tuple[int, ...],
        n_clbits: int,
    ) -> None:
        """Check quantum shape and engine-owned classical fields.

        Runtime initialization handles quantum conversion and per-shot copying.
        """
        classical = initial_state.classical
        capabilities = self._engine.capabilities
        if classical.clbits is not None:
            if not capabilities.supports_classical_register:
                raise BackendValidationError(
                    "engine does not support initial classical digits"
                )
            if len(classical.clbits) != n_clbits:
                raise BackendValidationError(
                    f"initial classical register has {len(classical.clbits)} digits; expected {n_clbits}"
                )
            if any(
                not isinstance(digit, (int, np.integer))
                or isinstance(digit, bool)
                or not 0 <= digit <= np.iinfo(np.int64).max
                for digit in classical.clbits
            ):
                raise BackendValidationError(
                    "initial classical digits must be non-negative int64 integers"
                )
        if classical.occupied is not None:
            if not capabilities.supports_occupancy:
                raise BackendValidationError(
                    f"method={self._state_field!r} cannot track carrier occupancy; "
                    "the selected engine has no occupancy state"
                )
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(system_dims)
                for index in classical.occupied
            ):
                raise BackendValidationError(
                    "initial occupancy contains an invalid subsystem index"
                )
        quantum = initial_state.quantum
        if quantum is None:
            return
        shape = getattr(quantum, "shape", None)
        if shape is None:
            shape = np.shape(quantum)
        shape = tuple(shape)
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
        if shape not in allowed:
            expected = " or ".join(str(expected_shape) for expected_shape in allowed)
            raise BackendValidationError(
                f"initial_state has shape {shape}, but this program has "
                f"subsystem dimensions {tuple(system_dims)}, so "
                f"method={self._state_field!r} needs {expected}"
            )

    def _validate_run_request(
        self, config: _SimulatorResultConfig, shots: int, facts: _PlanFacts
    ) -> _ResultRequest:
        """Resolve result fields and validate their shot requirements."""
        if config.counts is True and self._engine.capabilities.quantum.is_operator:
            raise BackendValidationError(
                f"method={self._state_field!r} cannot produce counts; it computes the "
                "program's operator rather than sampling outcomes from it"
            )
        stochastic = facts.stochastic_final_state
        counts, final_state = _resolve_result_flags(
            config,
            has_measurement=facts.has_measurement,
            stochastic_final_state=stochastic,
        )
        request = _ResultRequest(
            counts=counts,
            final_state=final_state,
            final_classical_state=config.final_classical_state,
        )
        requested_state = config.final_state is True

        # shots is only checked when the result actually depends on it: counts
        # always sample per shot, and a stochastic state export needs shots==1
        # below. A non-stochastic state-only request ignores shots entirely
        # (see the engine's per-shot path), so any value - including 0 - is fine.
        quantum = self._engine.capabilities.quantum
        common_stochastic = facts.has_measurement or (
            quantum.nonunitary_is_stochastic and (facts.has_reset or facts.has_channel)
        )
        if stochastic and (not common_stochastic or not facts.has_measurement):
            stochastic_sources = "stochastic execution"
        elif quantum.nonunitary_is_stochastic:
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
        if request.final_classical_state:
            if quantum.is_operator:
                raise BackendValidationError(
                    f"method={self._state_field!r} cannot produce a final classical "
                    "state; it computes the program's operator rather than evolving "
                    "a shot"
                )
            if stochastic and shots != 1:
                raise BackendValidationError(
                    f"final_classical_state with {stochastic_sources} is only "
                    "supported for shots == 1"
                )
        return request

    def _validate_engine_support(self, execution: _PreparedExecution) -> None:
        """Check the lowered program, controls and initial state against the engine."""
        facts = execution.facts
        simulation_config = execution.simulation
        capabilities = self._engine.capabilities
        # Engine execution support, such as kernel threads, fusion, and process shots.
        if (
            simulation_config.kernel_parallelism == "threads"
            and not capabilities.kernels.supports_kernel_threads
        ):
            raise BackendValidationError(
                "kernel_parallelism='threads' requires an engine with threaded "
                "numerical kernels"
            )
        if simulation_config.fusion and not capabilities.kernels.supports_fusion:
            raise BackendValidationError(
                "fusion=True is not supported by the selected matrix engine; fusion "
                "does not control compiled multi-shot execution"
            )
        if (
            simulation_config.shot_parallelism == "processes"
            and not capabilities.supports_process_shots
        ):
            raise BackendValidationError(
                "shot_parallelism='processes' is not supported by the selected engine"
            )
        # Engine method support, such as measurement, reset, channel noise, and classical register.
        method = self._state_field
        if capabilities.quantum.is_operator:
            if any(isinstance(step, CustomOperationStep) for step in execution.plan):
                raise UnsupportedOperationError(
                    "custom operations require statevector or density_matrix execution"
                )
        if facts.has_measurement and not capabilities.supports_classical_register:
            if not capabilities.quantum.is_operator:
                raise BackendValidationError(
                    f"method={method!r} cannot execute a measurement; the selected "
                    "engine has no classical register for reported outcomes"
                )
            raise BackendValidationError(
                f"method={method!r} cannot execute a measurement; it computes "
                "the program's operator, which no measurement outcome is part "
                "of (use method='statevector' or 'density_matrix' to sample "
                "outcomes)"
            )
        if facts.has_condition and not capabilities.supports_classical_register:
            raise BackendValidationError(
                f"method={method!r} cannot execute a feedforward condition; it "
                "has no classical register to evaluate one against"
            )
        if not capabilities.quantum.supports_nonunitary and (
            facts.has_reset or facts.has_channel
        ):
            source = "reset" if facts.has_reset else "channel noise"
            if not capabilities.quantum.is_operator:
                raise BackendValidationError(
                    f"method={method!r} cannot execute {source}; the selected "
                    "engine does not support non-unitary maps"
                )
            raise BackendValidationError(
                f"method={method!r} cannot execute {source}; a unitary cannot "
                "represent a non-unitary map (use method='superop' for the "
                "program's channel)"
            )

    # --- execution ---
    def _assemble_result(
        self,
        *,
        raw: RawResult[np.ndarray],
        config: _SimulatorResultConfig,
        simulation: _SimulationConfig,
        lowering: _LoweringContext,
        request: _ResultRequest,
        shots: int,
    ) -> Result:
        """Build one public result without depending on execution routing."""
        engine_allocation = lowering.engine_allocation
        classical_allocation = lowering.classical_allocation
        classical_dims = classical_allocation.classical_dims
        counts = None
        state = raw.state
        state_requested = request.final_state
        available: set[str] = set()
        if request.counts:
            counts = counts_dict_from_arrays(raw.outcome_keys, raw.outcome_counts)
            available.add("counts")
        if state_requested:
            available.add(self._state_field)
        if request.final_classical_state:
            available.add("classical_state")

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
                tuple(reversed(engine_allocation.device_operands)),
                lowering.resource_layout,
            )
        return Result(
            counts=counts,
            available=frozenset(available),
            classical_dims=classical_dims,
            data=extra_data,
            metadata=metadata,
            classical_state=raw.classical_state,
            **{self._state_field: state},
        )

    def _additional_result_data(
        self,
        *,
        config: _SimulatorResultConfig,
        simulation: _SimulationConfig,
        raw: RawResult[np.ndarray],
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

        A gate whose fields hold `Parameter`s lowers to an
        `ApplyParametricMatrixStep`; only sweeps lower such programs.
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
            elif isinstance(step, _AppliedOperation):
                if isinstance(step.operation, type(Barrier)):
                    continue
                if isinstance(step.operation, type(Reset)):
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
                        )
                    )

        return plan

    def _analyze_lowered_plan(
        self,
        plan: Sequence[ResolvedStep],
    ) -> _PlanFacts:
        """Translate one lowered plan into common execution semantics."""
        return self._analyze_common_plan_facts(plan)

    def _analyze_common_plan_facts(
        self,
        plan: Sequence[ResolvedStep],
        *,
        claimed_step_types: tuple[type[object], ...] = (),
    ) -> _PlanFacts:
        """Exhaustively derive runtime-independent matrix execution semantics."""
        quantum = self._engine.capabilities.quantum
        stochastic_final_state = False
        has_measurement = False
        has_reset = False
        has_channel = False
        has_condition = False

        for step in plan:
            if getattr(step, "condition", None) is not None:
                has_condition = True

            if claimed_step_types and isinstance(step, claimed_step_types):
                continue

            if isinstance(step, MeasurementStep):
                has_measurement = True
                stochastic_final_state = True
                continue

            if isinstance(step, (ApplyMatrixStep, ApplyParametricMatrixStep)):
                continue
            if isinstance(step, CustomOperationStep):
                # Arbitrary callbacks may sample and reset even under DM semantics.
                stochastic_final_state = True
                continue
            if isinstance(step, ResetStep):
                has_reset = True
                stochastic_final_state |= quantum.nonunitary_is_stochastic
            elif isinstance(step, ApplyChannelStep):
                has_channel = True
                stochastic_final_state |= quantum.nonunitary_is_stochastic
            else:
                raise TypeError(
                    f"unknown resolved execution step {type(step).__name__}"
                )

        return _PlanFacts(
            stochastic_final_state=stochastic_final_state,
            has_measurement=has_measurement,
            has_reset=has_reset,
            has_channel=has_channel,
            has_condition=has_condition,
        )

    def validate_noise_model(self, noise_model: NoiseModel) -> None:
        """Raise if this simulator cannot use a noise model.

        Matrix simulators accept operation-bound probability forms for
        supported built-in descriptors. Custom descriptors need a matching
        channel rule. Background sources, built-in damping descriptors in rate
        form, and ``ThermalRelaxation`` are rejected. For a matrix simulator,
        declare finite ``AmplitudeDamping`` or probability-form
        ``TransitionRelaxation`` and ``PhaseDamping`` channels instead. Only
        ``AtomArraySimulator`` accepts ``Loss``.

        This validates the model as a whole, not its selectors against a
        program, concrete target dimensions, or method-specific restrictions.

        Args:
            noise_model: The noise model to validate; it is not executed.

        Raises:
            BackendValidationError: If ``noise_model`` is not a ``NoiseModel``
                or contains unsupported declarations. The message lists every
                problem found.
        """
        if not isinstance(noise_model, NoiseModel):
            raise BackendValidationError("noise_model must be a NoiseModel")
        rejection_reasons = self._noise_model_rejection_reasons(noise_model)
        if rejection_reasons:
            raise BackendValidationError("; ".join(rejection_reasons))

    def _noise_model_rejection_reasons(
        self, noise_model: NoiseModel
    ) -> tuple[str, ...]:
        """Return model-level rejection reasons for an already typed model."""
        rejection_reasons: list[str] = []

        for channel, operation in noise_model._noise_sources():
            channel_type = type(channel)
            background = operation is None
            built_in_dual_form = channel_type in (
                AmplitudeDamping,
                TransitionRelaxation,
                PhaseDamping,
                Depolarizing,
            )
            rate_mode = built_in_dual_form and channel.rate is not None
            qualifiers: list[str] = []
            if built_in_dual_form:
                qualifiers.append("rate" if rate_mode else "p")
            if background:
                qualifiers.append("background")
            label = channel_type.__name__
            if qualifiers:
                label += f"({', '.join(qualifiers)})"
            if isinstance(channel, Loss):
                if not self._supports_loss:
                    rejection_reasons.append(
                        f"{label} is not supported: this backend does not model "
                        "carrier loss (use AtomArraySimulator)",
                    )
            elif background:
                rejection_reasons.append(
                    f"{label} is not supported: this matrix backend has no "
                    "continuous-time evolution model",
                )
            elif channel_type is ThermalRelaxation:
                rejection_reasons.append(
                    f"{label} is rejected by matrix-family policy because it "
                    "is a generator/time declaration; a registered channel "
                    "implementation does not override that policy. Declare "
                    "finite AmplitudeDamping or probability-form "
                    "TransitionRelaxation and PhaseDamping channels instead",
                )
            elif not self._impl_map.supports(channel_type):
                rejection_reasons.append(
                    f"{label} has no channel implementation on this backend",
                )
            elif rate_mode:
                rejection_reasons.append(
                    f"{label} is not supported: rate mode has no matrix-backend "
                    "Kraus implementation on this backend",
                )
        return tuple(dict.fromkeys(rejection_reasons))
