"""Model-neutral orchestration for pulse-emulator backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, final

import numpy as np

from ..._parameter_binding import _raise_for_unbound_parameters
from ..._index_allocation import (
    _ClassicalAllocation,
    _EngineAllocation,
    _describe_state_axes,
)
from ..._backends.backend_utils import (
    _normalize_config,
    _resolve_result_flags,
    _validate_result_shots,
)
from ..._backends.view_normalization import (
    ProgramInstruction,
    _break_grouped_operations,
)
from ...errors import (
    BackendExecutionError,
    BackendValidationError,
)
from ...job import Job
from ...noise import LindbladImplementationMap, NoiseModel, NoiseSupportReport
from ...operations import BarrierGate, Measurement, PulseOperation, ResetGate
from ...program import AppliedOperation, Program
from ...resource_layout import ResourceLayout
from ...result import (
    Result,
    _ResultConfig,
    counts_dict_from_arrays,
    reduce_to_counts,
)
from . import planning
from .engine import PulseEngine
from .config import _EmulatorConfig
from .planning import (
    PulsePlanFacts,
    PulsePlanStep,
    _PreparedPulseProgram,
    _PulseLoweringContext,
)
from .pulse import PulseImplementationMap
from .scheduling import _validate_schedule_mode
from .outcome import (
    ExecutionMode,
    _PulseExecutionSummary,
    _PulseResultRequest,
)
from .target import _PulseTarget


class _PulseBackend(ABC):
    """Private model-neutral pulse execution orchestration.

    This layer owns preparation, lowering, execution, propagation, error
    boundaries, and result assembly. A concrete family supplies only source
    validation, capability classification, execution-mode selection, runner
    creation, and optional metadata. Its bound target answers physical binding
    questions without adding backend hooks.
    """

    _coherent_execution_mode: ExecutionMode
    _target: _PulseTarget

    def __init__(
        self,
        model: object,
        *,
        noise: NoiseModel | None,
        gate_implementation_map: PulseImplementationMap,
        lindblad_implementation_map: LindbladImplementationMap,
    ) -> None:
        if noise is not None and not isinstance(noise, NoiseModel):
            raise BackendValidationError("noise must be a NoiseModel or None")
        if not isinstance(gate_implementation_map, PulseImplementationMap):
            raise BackendValidationError(
                "gate_implementation_map must be a PulseImplementationMap"
            )
        if not isinstance(lindblad_implementation_map, LindbladImplementationMap):
            raise BackendValidationError(
                "lindblad_implementation_map must be a LindbladImplementationMap"
            )
        self._model = model
        self._gate_implementation_map = gate_implementation_map.copy()
        self._lindblad_implementation_map = lindblad_implementation_map.copy()
        source_noise = NoiseModel() if noise is None else noise
        self._noise_model = source_noise._copy()

    @property
    @final
    def model(self) -> object:
        """Return the exact immutable physics model retained by the backend."""
        return self._model

    @final
    def _set_target(self, target: _PulseTarget) -> None:
        """Install the one target after family construction-time checks."""
        if hasattr(self, "_target"):
            raise BackendValidationError("pulse backend target is already installed")
        if target.model is not self._model:
            raise BackendValidationError(
                "pulse backend target must retain the exact backend model"
            )
        self._target = target

    @final
    def _require_captured_noise_support(self) -> None:
        """Reject an unsupported captured model before target construction."""
        report = self.check_noise_support(self._noise_model)
        if not report.supported:
            raise BackendValidationError("; ".join(report.warnings))

    @final
    def _prepare_program(
        self,
        program: Program,
        resource_layout: ResourceLayout | None = None,
    ) -> _PreparedPulseProgram:
        """Produce the sole immutable preparation value for one invocation."""
        if not isinstance(program, Program):
            raise BackendValidationError("program must be a Program")
        self._validate_source_program(program)
        resolved_layout = (
            self._target.bind_program(program)
            if resource_layout is None
            else self._target.bind_program(program, resource_layout)
        )
        engine_allocation = _EngineAllocation(
            tuple(self._target.device_labels),
            tuple(
                self._target.local_dimension
                for _device_operand in self._target.device_labels
            ),
        )
        classical_allocation = _ClassicalAllocation.from_program(program)
        self._noise_model._validate_for(program, frozenset(self._target.device_labels))
        context = _PulseLoweringContext(
            resource_layout=resolved_layout,
            engine_allocation=engine_allocation,
            classical_allocation=classical_allocation,
        )
        operations = _break_grouped_operations(program.operations)
        plan = tuple(self._lower(operations, context))
        background_noise = planning._resolve_background_noise(
            target=self._target,
            resource_layout=resolved_layout,
            engine_allocation=engine_allocation,
            noise_model=self._noise_model,
            implementation_map=self._lindblad_implementation_map,
        )
        supported_background = any(
            operation is None
            and self._lindblad_implementation_map.get(type(channel)) is not None
            for channel, operation in self._noise_model._noise_sources()
        )
        facts = planning._derive_plan_facts(
            plan,
            background_noise,
            has_supported_background_lindblad_registration=supported_background,
        )
        return _PreparedPulseProgram(
            plan=plan,
            facts=facts,
            resource_layout=resolved_layout,
            engine_allocation=engine_allocation,
            classical_allocation=classical_allocation,
            background_noise=background_noise,
        )

    @final
    def _lower(
        self,
        operations: Sequence[ProgramInstruction],
        context: _PulseLoweringContext,
    ) -> list[PulsePlanStep]:
        """Lower scalar instructions into an ordered, unplaced pulse plan."""
        resource_layout = context.resource_layout
        engine_allocation = context.engine_allocation
        classical_allocation = context.classical_allocation
        plan: list[PulsePlanStep] = []
        for step in operations:
            if isinstance(step, Measurement):
                reported_digit_maps = tuple(
                    self._target.reported_digit_map(
                        resource_layout.device_label(target)
                    )
                    for target in step.targets
                )
                plan.append(
                    planning._lower_measurement(
                        step,
                        reported_digit_maps,
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
                    if isinstance(step.operation, PulseOperation):
                        plan.append(
                            planning._lower_direct(
                                step,
                                target=self._target,
                                context=context,
                            )
                        )
                        continue
                    plan.append(
                        planning._lower_gate(
                            step,
                            target=self._target,
                            context=context,
                            gate_implementation_map=self._gate_implementation_map,
                            noise_model=self._noise_model,
                            lindblad_implementation_map=(
                                self._lindblad_implementation_map
                            ),
                        )
                    )
        return plan

    @final
    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        resource_layout: ResourceLayout | None = None,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job:
        """Validate, execute, and package one pulse-program run.

        ``simulation_config`` accepts ``seed`` and ``schedule_mode``, the only
        two controls pulse execution honors. ``schedule_mode`` is ``"ASAP"``
        by default and may be ``"ALAP"``; both are lightweight placement
        policies over dependencies and claimed physical resources, not
        compiler-produced hardware schedules. The matrix backend's
        ``shot_parallelism``, ``kernel_parallelism``, ``max_workers``, and
        ``fusion`` are rejected here:
        pulse execution is one serial solver call with no engine those settings
        could steer.

        ``result_config`` accepts ``counts`` and ``final_state``. When omitted,
        counts default on for programs containing measurement and the final
        state defaults on for programs without measurement. The selected
        model family and execution mode determine whether that state is a
        statevector or density matrix. A measured final state is one sampled
        posterior and therefore requires ``shots == 1``.

        Each pulse emulator constructs its model family's fixed product
        initial state for every run. TransmonEmulator and Atom3LevelEmulator
        use local level 0; Atom2LevelEmulator uses its ground level. This
        method has no initial_state argument.

        Args:
            program: Program to bind, lower, and execute.
            shots: Number of repetitions used when counts are requested.
            resource_layout: Optional public program-to-device mapping. The
                target supplies its default binding when omitted.
            simulation_config: Optional pulse-execution settings. Unknown or
                incompatible keys are rejected before execution.
            result_config: Optional result request. Unknown or incompatible
                keys are rejected before execution.

        Returns:
            An eager terminal :class:`~fatqat.Job`. Validation and lowering
            failures raise directly. Solver-stage failures produce a failed
            job whose :meth:`~fatqat.Job.result` raises
            :class:`~fatqat.errors.BackendExecutionError`.

        Raises:
            BackendValidationError: If the target cannot bind the program,
                noise/configuration is unsupported, or requested results are
                incompatible with ``shots`` and measurement.
            UnsupportedOperationError: If no pulse implementation exists for
                an operation family or its ordered device operands.
            PulseImplementationError: If a selected custom pulse rule fails
                unexpectedly or returns the wrong value type.
        """
        simulation = _normalize_config(
            simulation_config,
            _EmulatorConfig,
            "simulation_config",
            backend_name=self._backend_name(),
        )
        result = _normalize_config(
            result_config,
            _ResultConfig,
            "result_config",
            backend_name=self._backend_name(),
        )
        _raise_for_unbound_parameters(program.operations)
        prepared = self._prepare_program(program, resource_layout)
        request = self._validate(
            result,
            shots,
            prepared.facts,
        )
        try:
            return Job(
                status="DONE",
                result=self._execute(prepared, request, simulation, shots),
            )
        except Exception as exc:  # execution failures belong on the eager Job
            # The public message stays stable and free of solver internals,
            # but the original exception is chained so a developer (and a
            # traceback) can still see what actually failed. Assigning
            # `__cause__` rather than raising keeps this an eager failed Job.
            failure = BackendExecutionError("Pulse backend execution failed")
            failure.__cause__ = exc
            return Job(status="ERROR", error=failure)

    @final
    def propagator(
        self,
        program: Program,
        *,
        apply_final_frame: bool = True,
        schedule_mode: Literal["ASAP", "ALAP"] = "ASAP",
        resource_layout: ResourceLayout | None = None,
    ) -> np.ndarray:
        """Return the coherent full-model propagator for ``program``.

        For Hilbert-space dimension D, the result contains D * D complex
        entries. Time-dependent propagation evolves the full operator, so use
        this method for small models or occasional operator construction, not
        repeated large-system time-series sweeps.

        Intermediate virtual-frame updates always rotate later phase-sensitive
        controls. By default the remaining terminal frame transformation is
        also composed onto the returned propagator; set
        ``apply_final_frame=False`` to inspect Hamiltonian-generated evolution
        before that final basis transformation.

        The result is a complex NumPy array over the selected model family's
        full physical Hilbert space. Its rows and columns use the same
        canonical little-endian basis order as returned states: physical axis
        0 is the least-significant subsystem. Measurement, reset, and classical
        conditions are rejected. Bound collapse terms are rejected when the
        plan contains nonzero elapsed evolution; rate-based noise has no effect
        on an empty or frame-only plan because no time elapses.

        Args:
            program: Coherent program to lower, schedule, and propagate.
            apply_final_frame: Whether to compose the terminal virtual-frame
                transformation. Intermediate frame updates are always honored.
            schedule_mode: Lightweight pulse placement policy, ``"ASAP"`` or
                ``"ALAP"``.
            resource_layout: Optional public program-to-device mapping. The
                target supplies its default binding when omitted.

        Returns:
            Full-model coherent propagator as a NumPy array.
        """
        if type(apply_final_frame) is not bool:
            raise BackendValidationError("apply_final_frame must be a bool")
        schedule_mode = _validate_schedule_mode(schedule_mode)
        _raise_for_unbound_parameters(program.operations)
        prepared = self._prepare_program(program, resource_layout)

        if not prepared.plan:
            return np.eye(self._target.hilbert_dimension, dtype=complex)

        self._validate_propagator_facts(prepared.facts)
        runner = self._create_runner(
            prepared,
            execution_mode=self._coherent_execution_mode,
            retain_final_state=True,
        )
        engine = PulseEngine(runner, schedule_mode=schedule_mode)
        try:
            return np.asarray(
                engine.propagator(
                    prepared.plan,
                    apply_final_frame=apply_final_frame,
                ).full(),
                dtype=complex,
            )
        except BackendValidationError:
            raise
        except Exception as exc:
            raise BackendExecutionError("Pulse propagator construction failed") from exc

    @staticmethod
    @final
    def _validate_propagator_facts(facts: PulsePlanFacts) -> None:
        if facts.has_measurement:
            raise BackendValidationError("propagator does not support measurement")
        if facts.has_reset:
            raise BackendValidationError("propagator does not support reset")
        if facts.has_conditions:
            raise BackendValidationError(
                "propagator does not support classically conditioned operations"
            )
        if facts.has_nonzero_evolution and facts.has_resolved_lindblad:
            raise BackendValidationError(
                "propagator does not support dissipative Lindblad evolution"
            )

    @final
    def _validate(
        self,
        config: _ResultConfig,
        shots: int,
        facts: PulsePlanFacts,
    ) -> _PulseResultRequest:
        """Resolve default output requests and validate their shot constraints."""
        counts, final_state = _resolve_result_flags(
            config,
            has_measurement=facts.has_measurement,
            stochastic_final_state=facts.has_measurement,
        )
        execution_mode = self._resolve_execution_mode(facts)
        _validate_result_shots(
            counts=counts,
            explicit_final_state=config.final_state is True,
            stochastic_final_state=facts.has_measurement,
            shots=shots,
            shots_type_error=(
                "shots must be an int when requested results depend on it"
            ),
            state_label=self._state_label_for_execution_mode(execution_mode),
            stochastic_sources="physical measurement sampling",
        )
        return _PulseResultRequest(
            counts=counts,
            final_state=final_state,
            execution_mode=execution_mode,
        )

    @final
    def _execute(
        self,
        prepared: _PreparedPulseProgram,
        request: _PulseResultRequest,
        simulation: _EmulatorConfig,
        shots: int,
    ) -> Result:
        """Execute a validated plan and convert private shot payloads to Result."""
        runner = self._create_runner(
            prepared,
            execution_mode=request.execution_mode,
            retain_final_state=request.final_state,
        )
        execution_shots = shots if request.counts else 1
        engine = PulseEngine(runner, schedule_mode=simulation.schedule_mode)
        engine_method = (
            engine.run_terminal_trajectory_batch
            if request.execution_mode == "trajectory"
            else engine.run
        )
        outcomes = engine_method(
            prepared.plan,
            shots=execution_shots,
            n_clbits=prepared.classical_allocation.n_clbits,
            rng=np.random.default_rng(simulation.seed),
        )
        summary = self._summarize_execution(
            outcomes,
            require_final_state=request.final_state,
            runner=runner,
        )
        return self._assemble_result(
            prepared,
            request,
            simulation,
            shots,
            summary,
        )

    @staticmethod
    @final
    def _summarize_execution(
        outcomes: tuple[Any, ...],
        *,
        require_final_state: bool,
        runner: Any,
    ) -> _PulseExecutionSummary:
        if not outcomes:
            raise BackendExecutionError("pulse execution produced no shot outcomes")
        kinds = {outcome.final_state_kind for outcome in outcomes}
        if len(kinds) != 1:
            raise BackendExecutionError(
                "pulse execution produced mixed final-state kinds"
            )
        final_state_kind = kinds.pop()
        if final_state_kind not in ("statevector", "density_matrix"):
            raise BackendExecutionError(
                "pulse execution produced unknown final-state kind "
                f"{final_state_kind!r}"
            )
        if require_final_state and any(
            outcome.final_state is None for outcome in outcomes
        ):
            raise BackendExecutionError(
                "pulse execution omitted a requested final state"
            )
        return _PulseExecutionSummary(
            outcomes=outcomes,
            final_state_kind=final_state_kind,
            solver_metadata=runner.solver_metadata(),
        )

    @final
    def _assemble_result(
        self,
        prepared: _PreparedPulseProgram,
        request: _PulseResultRequest,
        simulation: _EmulatorConfig,
        shots: int,
        summary: _PulseExecutionSummary,
    ) -> Result:
        final_state = summary.outcomes[-1].final_state
        counts = None
        available = set()
        if request.counts:
            keys, values = reduce_to_counts(
                [outcome.classical_digits for outcome in summary.outcomes]
            )
            counts = counts_dict_from_arrays(keys, values)
            available.add("counts")
        statevector = None
        density_matrix = None
        if request.final_state:
            available.add(summary.final_state_kind)
            if summary.final_state_kind == "statevector":
                statevector = final_state
            else:
                density_matrix = final_state
        metadata = {
            "backend_name": self._backend_name(),
            "shots": shots,
            "simulation_config": asdict(simulation),
            "result_config": {
                "counts": request.counts,
                "final_state": request.final_state,
            },
            "solver": dict(summary.solver_metadata),
        }
        if request.final_state:
            metadata["state_axes"] = _describe_state_axes(
                prepared.engine_allocation,
                prepared.resource_layout,
            )
        return Result(
            counts=counts,
            statevector=statevector,
            density_matrix=density_matrix,
            available=frozenset(available),
            classical_dims=prepared.classical_allocation.classical_dims,
            metadata=metadata,
        )

    @abstractmethod
    def _resolve_execution_mode(
        self,
        facts: PulsePlanFacts,
    ) -> ExecutionMode:
        """Select the private representation from plan and noise physics."""
        raise NotImplementedError

    @staticmethod
    @final
    def _state_label_for_execution_mode(execution_mode: ExecutionMode) -> str:
        """Return the public result label used by shot validation."""
        return "density_matrix" if execution_mode == "density_matrix" else "statevector"

    @final
    def _backend_name(self) -> str:
        """Return the public backend name used in errors and metadata."""
        return type(self).__name__

    def _validate_source_program(self, program: Program) -> None:
        """Validate only family-specific source-language restrictions."""
        del program

    @abstractmethod
    def _classify_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report the family capability for one already typed noise model."""
        raise NotImplementedError

    @abstractmethod
    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        """Create a runner from already-bound execution data."""
        del prepared, execution_mode, retain_final_state
        raise NotImplementedError

    @final
    def check_noise_support(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report whether this backend can execute an explicit noise model.

        This check is program-agnostic. It classifies source types, authored
        parameter modes, background support, loss support, and readout
        support. Program references and physical selectors are validated when
        a concrete program and resource layout are prepared.

        Args:
            noise_model: Noise model to inspect without executing a program.

        Returns:
            A frozen report naming accepted and rejected sources.

        Raises:
            BackendValidationError: If ``noise_model`` is not a
                :class:`~fatqat.NoiseModel`.
        """
        if not isinstance(noise_model, NoiseModel):
            raise BackendValidationError("noise_model must be a NoiseModel")
        return self._classify_noise(noise_model)
