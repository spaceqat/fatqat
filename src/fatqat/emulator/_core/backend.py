"""Model-neutral orchestration for pulse-emulator backends."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, final

import numpy as np

from ..._expectation import (
    _ExpectationExecution,
    _TermOccurrence,
    _combine_term_statistics,
    _plan_term_occurrences,
    _reduce_outcome_counts,
    expectation_density_matrix,
    expectation_statevector,
)
from ..._parameter_binding import _raise_for_unbound_parameters
from ..._index_allocation import (
    _ClassicalAllocation,
    _EngineAllocation,
    _describe_state_axes,
)
from ..._backends.backend_utils import (
    _canonicalize_method,
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
    UnsupportedOperationError,
)
from ...job import Job
from ...noise import NoiseModel
from ...noise.lindblad import LindbladImplementationMap
from ...observable import Observable
from ...operations import Barrier, Measurement, PulseOperation, Reset
from ...program import Program, _AppliedOperation
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
from .outcome import (
    ExecutionMode,
    _PulseExecutionSummary,
    _PulseResultRequest,
)
from .target import _PulseTarget


@dataclass(frozen=True, slots=True)
class _BoundPulseExpectation:
    """One logical term occurrence bound to pulse-engine resources."""

    occurrence: _TermOccurrence
    engine_factors: tuple[tuple[int, str], ...]
    scratch_indices: tuple[int, ...]
    measurement: PulsePlanStep | None


@dataclass(frozen=True, slots=True)
class _PreparedPulseExpectation:
    """Validated pulse expectation work before numerical execution."""

    simulation: _EmulatorConfig
    program: _PreparedPulseProgram
    constants: tuple[float, ...]
    bound_occurrences: tuple[_BoundPulseExpectation, ...]
    execution_mode: ExecutionMode


class _PulseBackend(ABC):
    """Private model-neutral pulse execution orchestration.

    This layer owns preparation, lowering, execution, error boundaries, and
    result assembly. A concrete family supplies only source and noise-model
    validation, runner creation, and optional metadata. Its bound target
    answers physical binding questions without adding backend hooks.
    """

    _target: _PulseTarget
    _runtime_name: str = "qutip"

    # Families may replace this with a dataclass derived from
    # ``_EmulatorConfig`` for family-specific per-run controls. The normalizer
    # derives accepted keys from the schema, so other families reject them.
    _simulation_config_cls: type[_EmulatorConfig] = _EmulatorConfig

    def __init__(
        self,
        model: object,
        *,
        method: object,
        noise: NoiseModel | None,
        gate_implementation_map: PulseImplementationMap,
        lindblad_implementation_map: LindbladImplementationMap,
    ) -> None:
        canonical_method = _canonicalize_method(
            method, {"statevector", "density_matrix", "unitary"}
        )
        if canonical_method is None:
            raise BackendValidationError(
                "method must be 'statevector', 'density_matrix', or 'unitary' "
                "(aliases 'SV' and 'DM' are accepted)"
            )
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
        self._method = canonical_method
        self._gate_implementation_map = gate_implementation_map.copy()
        self._lindblad_implementation_map = lindblad_implementation_map.copy()
        source_noise = NoiseModel() if noise is None else noise
        self._noise_model = source_noise._copy()

    @property
    @final
    def model(self) -> object:
        """Return this emulator's physics model."""
        return self._model

    @property
    @final
    def method(self) -> str:
        """Return the canonical mathematical representation for this emulator."""

        return self._method

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
        operations = _break_grouped_operations(program._instructions)
        plan = tuple(self._lower(operations, context))
        background_noise = planning._resolve_background_noise(
            target=self._target,
            resource_layout=resolved_layout,
            engine_allocation=engine_allocation,
            noise_model=self._noise_model,
            implementation_map=self._lindblad_implementation_map,
        )
        facts = planning._derive_plan_facts(plan, background_noise)
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
    ) -> Job[Result]:
        """Run a program on this pulse emulator.

        The emulator family fixes the initial product state; pulse backends do
        not accept an ``initial_state`` argument.

        Args:
            program: Program to execute.
            shots: Number of repetitions used when counts are requested.
                Counts require a positive built-in ``int``. A stochastic
                final state requires ``shots == 1``.
            resource_layout: Optional program-to-device mapping. The emulator
                uses its default mapping when omitted.
            simulation_config: Optional ``dict`` of pulse-execution settings.
                Every pulse emulator accepts:

                - ``"seed"`` (``int | None``, default ``None``): Seed
                  stochastic sampling. Use a non-negative integer; ``None``
                  uses fresh entropy, and booleans are rejected.
                - ``"schedule_mode"`` (``str: "ASAP" or "ALAP"``, default
                  ``"ASAP"``): Choose whether operations are placed as early
                  or as late as possible while preserving dependencies and
                  physical-resource conflicts.

                ``Atom2LevelEmulator`` additionally accepts
                ``"interaction_cutoff"`` (finite nonnegative ``Real`` or
                ``None``, default ``None``) to truncate that run's static
                ``C6/R^6`` interaction Hamiltonian by pair distance.

                Other keys, including matrix-only execution settings, are
                rejected.
            result_config: Optional ``dict`` of output requests. Accepted keys
                are:

                - ``"counts"`` (``bool | None``, default ``None``):
                  ``True`` requests classical counts, ``False`` suppresses
                  them, and ``None`` enables them when measurement exists.
                  Counts require a positive integer ``shots`` value.
                - ``"final_state"`` (``bool | None``, default ``None``):
                  ``True`` requests the method-native ``statevector``,
                  ``density_matrix``, or ``unitary``; ``False`` suppresses it;
                  ``None`` enables deterministic unmeasured output. A
                  stochastic final state requires an explicit request and
                  ``shots == 1``.

                Other keys are rejected.

        Returns:
            A terminal ``Job``. Validation errors are raised by ``run()``.
            Execution errors produce a failed job whose ``result()`` raises
            ``BackendExecutionError``.

        Raises:
            BackendValidationError: If the program, noise, or configuration
                is unsupported, or if requested results are incompatible with
                ``shots`` and measurement.
            TypeError: If ``simulation_config`` or ``result_config`` is not a
                ``dict`` or ``None``.
            UnsupportedOperationError: If no pulse implementation exists for
                an operation family or its ordered device operands.
            PulseImplementationError: If a selected custom pulse rule fails
                unexpectedly or returns the wrong value type.
        """
        simulation = _normalize_config(
            simulation_config,
            self._simulation_config_cls,
            "simulation_config",
            backend_name=self._backend_name(),
        )
        result = _normalize_config(
            result_config,
            _ResultConfig,
            "result_config",
            backend_name=self._backend_name(),
        )
        _raise_for_unbound_parameters(program._instructions)
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
    def _validate(
        self,
        config: _ResultConfig,
        shots: int,
        facts: PulsePlanFacts,
    ) -> _PulseResultRequest:
        """Resolve default output requests and validate their shot constraints."""
        stochastic_final_state = (
            (self._method == "statevector" and facts.has_potentially_active_lindblad)
            or facts.has_measurement
            or (self._method == "statevector" and facts.has_reset)
        )
        counts, final_state = _resolve_result_flags(
            config,
            has_measurement=facts.has_measurement,
            stochastic_final_state=stochastic_final_state,
        )
        self._validate_method_facts(facts, counts=counts)
        execution_mode = self._execution_mode(facts)
        _validate_result_shots(
            counts=counts,
            explicit_final_state=config.final_state is True,
            stochastic_final_state=stochastic_final_state,
            shots=shots,
            shots_type_error=(
                "shots must be an int when requested results depend on it"
            ),
            state_label=self._method,
            stochastic_sources=self._stochastic_sources(facts),
        )
        return _PulseResultRequest(
            counts=counts,
            final_state=final_state,
            method=self._method,
            execution_mode=execution_mode,
        )

    @final
    def _stochastic_sources(self, facts: PulsePlanFacts) -> str:
        sources = []
        if facts.has_measurement:
            sources.append("physical measurement sampling")
        if self._method == "statevector" and facts.has_reset:
            sources.append("reset sampling")
        if self._method == "statevector" and facts.has_potentially_active_lindblad:
            sources.append("trajectory sampling")
        return ", ".join(sources)

    @final
    def _execution_mode(self, facts: PulsePlanFacts) -> ExecutionMode | None:
        if self._method == "unitary":
            return None
        if self._method == "density_matrix":
            return "density_matrix"
        if facts.has_potentially_active_lindblad:
            return "trajectory"
        return "statevector"

    @final
    def _validate_method_facts(self, facts: PulsePlanFacts, *, counts: bool) -> None:
        if self._method != "unitary":
            return
        if facts.has_measurement:
            raise BackendValidationError("unitary method does not support measurement")
        if facts.has_reset:
            raise BackendValidationError("unitary method does not support reset")
        if facts.has_conditions:
            raise BackendValidationError(
                "unitary method does not support classically conditioned operations"
            )
        if facts.has_potentially_active_lindblad:
            raise BackendValidationError(
                "unitary method does not support dissipative Lindblad evolution"
            )
        if counts:
            raise BackendValidationError("unitary method does not support counts")

    @final
    def _execute(
        self,
        prepared: _PreparedPulseProgram,
        request: _PulseResultRequest,
        simulation: _EmulatorConfig,
        shots: int,
    ) -> Result:
        """Execute a validated plan and convert private shot payloads to Result."""
        if request.method == "unitary":
            return self._execute_unitary(prepared, request, simulation, shots)
        if request.execution_mode is None:
            raise BackendExecutionError("state execution requires an execution mode")
        execution_shots = shots if request.counts else 1
        summary = self._execute_state(
            prepared,
            plan=prepared.plan,
            execution_mode=request.execution_mode,
            simulation=simulation,
            shots=execution_shots,
            n_clbits=prepared.classical_allocation.n_clbits,
            rng=np.random.default_rng(simulation.seed),
            retain_final_state=request.final_state,
        )
        return self._assemble_result(
            prepared,
            request,
            simulation,
            shots,
            summary,
        )

    @final
    def _execute_state(
        self,
        prepared: _PreparedPulseProgram,
        *,
        plan: tuple[PulsePlanStep, ...],
        execution_mode: ExecutionMode,
        simulation: _EmulatorConfig,
        shots: int,
        n_clbits: int,
        rng: np.random.Generator,
        retain_final_state: bool,
    ) -> _PulseExecutionSummary:
        """Execute one state plan through the normal pulse runner and engine."""
        runner = self._create_runner(
            prepared,
            simulation=simulation,
            execution_mode=execution_mode,
            retain_final_state=retain_final_state,
        )
        engine = PulseEngine(runner, schedule_mode=simulation.schedule_mode)
        engine_method = (
            engine.run_trajectories if execution_mode == "trajectory" else engine.run
        )
        outcomes = engine_method(
            plan,
            shots=shots,
            n_clbits=n_clbits,
            rng=rng,
        )
        return self._summarize_execution(
            outcomes,
            require_final_state=retain_final_state,
            runner=runner,
        )

    @final
    def _run_expectation(
        self,
        program: Program,
        observables: tuple[Observable, ...],
        *,
        shots: int,
        simulation_config: dict[str, Any] | None,
    ) -> Job[_ExpectationExecution]:
        """Execute one private exact or sampled pulse expectation request."""
        prepared = self._prepare_expectation(
            program,
            observables,
            shots=shots,
            simulation_config=simulation_config,
        )
        metadata = {
            "backend_name": self._backend_name(),
            "method": self._method,
            "runtime": self._runtime_name,
        }
        try:
            runtime_details = None
            if not prepared.bound_occurrences:
                values = prepared.constants
                standard_errors = (0.0,) * len(values)
            elif shots == 0:
                values, runtime_details = self._execute_exact_expectation(prepared)
                standard_errors = (0.0,) * len(values)
            else:
                (
                    values,
                    standard_errors,
                    runtime_details,
                ) = self._execute_sampled_expectation(prepared, shots=shots)
            if runtime_details is not None:
                metadata["runtime_details"] = dict(runtime_details)
            return Job(
                status="DONE",
                result=_ExpectationExecution(values, standard_errors, metadata),
            )
        except Exception as exc:
            failure = BackendExecutionError("Pulse backend execution failed")
            failure.__cause__ = exc
            return Job(status="ERROR", error=failure)

    def _prepare_expectation(
        self,
        program: Program,
        observables: tuple[Observable, ...],
        *,
        shots: int,
        simulation_config: dict[str, Any] | None,
    ) -> _PreparedPulseExpectation:
        """Validate, lower, and bind one pulse expectation request."""
        simulation = _normalize_config(
            simulation_config,
            self._simulation_config_cls,
            "simulation_config",
            backend_name=self._backend_name(),
        )
        prepared = self._prepare_program(program)
        if self._method == "unitary":
            raise UnsupportedOperationError(
                "pulse unitary execution computes an operator, not a state whose "
                "expectation can be evaluated"
            )
        unsupported_dimensions = tuple(
            dict.fromkeys(
                dimension
                for dimension in prepared.engine_allocation.system_dims
                if dimension != 2
            )
        )
        if unsupported_dimensions:
            raise UnsupportedOperationError(
                f"{self._backend_name()} has no defined qubit-Observable embedding "
                f"for local dimensions {unsupported_dimensions!r}"
            )
        if prepared.facts.has_reset:
            raise UnsupportedOperationError(
                "pulse expectation execution does not support reset"
            )
        if (
            shots == 0
            and self._method == "statevector"
            and prepared.facts.has_potentially_active_lindblad
        ):
            raise UnsupportedOperationError(
                "an exact statevector expectation is unavailable for potentially "
                "active Lindblad evolution"
            )
        execution_mode = self._execution_mode(prepared.facts)
        if execution_mode is None:
            raise UnsupportedOperationError(
                "pulse expectation execution requires a state method"
            )

        occurrences, constants = _plan_term_occurrences(observables)
        bound_occurrences = self._bind_expectation_occurrences(
            program,
            prepared,
            occurrences,
            shots=shots,
        )
        return _PreparedPulseExpectation(
            simulation=simulation,
            program=prepared,
            constants=constants,
            bound_occurrences=bound_occurrences,
            execution_mode=execution_mode,
        )

    def _bind_expectation_occurrences(
        self,
        program: Program,
        prepared: _PreparedPulseProgram,
        occurrences: tuple[_TermOccurrence, ...],
        *,
        shots: int,
    ) -> tuple[_BoundPulseExpectation, ...]:
        """Bind logical Pauli factors to pulse-engine resources and readout."""
        logical_refs = tuple(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        scratch_start = prepared.classical_allocation.n_clbits
        bound_occurrences = []
        for occurrence in occurrences:
            factor_refs = tuple(
                logical_refs[index] for index, _letter in occurrence.logical_factors
            )
            engine_factors = tuple(
                (
                    prepared.engine_allocation.engine_index(
                        prepared.resource_layout.device_label(ref)
                    ),
                    letter,
                )
                for ref, (_index, letter) in zip(
                    factor_refs,
                    occurrence.logical_factors,
                    strict=True,
                )
            )
            if shots > 0 and any(
                letter in {"X", "Y"} for _index, letter in engine_factors
            ):
                raise UnsupportedOperationError(
                    "sampled pulse expectations do not support X or Y basis factors"
                )

            reported_digit_maps = tuple(
                self._target.reported_digit_map(
                    prepared.resource_layout.device_label(ref)
                )
                for ref in factor_refs
            )
            if shots == 0:
                if (
                    planning._expectation_confusions(
                        factor_refs,
                        reported_digit_maps,
                        prepared.resource_layout,
                        self._noise_model,
                    )
                    is not None
                ):
                    raise UnsupportedOperationError(
                        "an exact pulse expectation cannot apply selected readout "
                        "confusion"
                    )
                measurement = None
            else:
                measurement = planning._build_expectation_measurement(
                    factor_refs,
                    tuple(index for index, _letter in engine_factors),
                    scratch_start=scratch_start,
                    target=self._target,
                    resource_layout=prepared.resource_layout,
                    noise_model=self._noise_model,
                )
            bound_occurrences.append(
                _BoundPulseExpectation(
                    occurrence=occurrence,
                    engine_factors=engine_factors,
                    scratch_indices=tuple(
                        range(scratch_start, scratch_start + len(engine_factors))
                    ),
                    measurement=measurement,
                )
            )
        return tuple(bound_occurrences)

    def _execute_exact_expectation(
        self,
        prepared: _PreparedPulseExpectation,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        """Contract every observable against one pulse-evolved state."""
        program = prepared.program
        summary = self._execute_state(
            program,
            plan=program.plan,
            execution_mode=prepared.execution_mode,
            simulation=prepared.simulation,
            shots=1,
            n_clbits=program.classical_allocation.n_clbits,
            rng=np.random.default_rng(prepared.simulation.seed),
            retain_final_state=True,
        )
        state = summary.outcomes[-1].final_state
        if state is None:
            raise BackendExecutionError(
                "pulse expectation execution omitted its final state"
            )
        kernel = (
            expectation_statevector
            if self._method == "statevector"
            else expectation_density_matrix
        )
        terms_by_observable = [[] for _constant in prepared.constants]
        for bound in prepared.bound_occurrences:
            terms_by_observable[bound.occurrence.observable_index].append(
                (bound.occurrence.coefficient, bound.engine_factors)
            )
        values = tuple(
            constant + kernel(state, tuple(terms))
            for constant, terms in zip(
                prepared.constants,
                terms_by_observable,
                strict=True,
            )
        )
        return values, dict(summary.runtime_details)

    def _execute_sampled_expectation(
        self,
        prepared: _PreparedPulseExpectation,
        *,
        shots: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
        """Execute native pulse outcomes for each diagonal term occurrence."""
        statistics = []
        runtime_details = None
        child_seeds = np.random.SeedSequence(prepared.simulation.seed).spawn(
            len(prepared.bound_occurrences)
        )
        for child_seed, bound in zip(
            child_seeds,
            prepared.bound_occurrences,
            strict=True,
        ):
            if bound.measurement is None:
                raise RuntimeError("sampled pulse expectation has no measurement")
            summary = self._execute_state(
                prepared.program,
                plan=prepared.program.plan + (bound.measurement,),
                execution_mode=prepared.execution_mode,
                simulation=prepared.simulation,
                shots=shots,
                n_clbits=(
                    prepared.program.classical_allocation.n_clbits
                    + len(bound.engine_factors)
                ),
                rng=np.random.default_rng(child_seed),
                retain_final_state=False,
            )
            runtime_details = dict(summary.runtime_details)
            statistics.append(
                _reduce_outcome_counts(
                    bound.engine_factors,
                    (
                        (
                            tuple(
                                outcome.classical_digits[index]
                                for index in bound.scratch_indices
                            ),
                            1,
                        )
                        for outcome in summary.outcomes
                    ),
                )
            )
        values, standard_errors = _combine_term_statistics(
            prepared.constants,
            tuple(bound.occurrence for bound in prepared.bound_occurrences),
            statistics,
            shots=shots,
        )
        if runtime_details is None:
            raise RuntimeError("sampled pulse expectation produced no execution")
        return values, standard_errors, runtime_details

    @final
    def _execute_unitary(
        self,
        prepared: _PreparedPulseProgram,
        request: _PulseResultRequest,
        simulation: _EmulatorConfig,
        shots: int,
    ) -> Result:
        unitary = None
        runtime_details = None
        available = frozenset()
        if request.final_state:
            if prepared.plan:
                runner = self._create_runner(
                    prepared,
                    simulation=simulation,
                    execution_mode="statevector",
                    retain_final_state=False,
                )
                engine = PulseEngine(runner, schedule_mode=simulation.schedule_mode)
                unitary = np.asarray(
                    engine.propagator(prepared.plan, apply_final_frame=True).full(),
                    dtype=complex,
                )
                runtime_details = runner.runtime_details()
            else:
                unitary = np.eye(self._target.hilbert_dimension, dtype=complex)
            available = frozenset({"unitary"})
        metadata = self._result_metadata(request, simulation, shots)
        if runtime_details is not None:
            metadata["runtime_details"] = dict(runtime_details)
        if request.final_state:
            metadata["state_axes"] = _describe_state_axes(
                prepared.engine_allocation.device_operands,
                prepared.resource_layout,
            )
        return Result(
            unitary=unitary,
            available=available,
            classical_dims=prepared.classical_allocation.classical_dims,
            metadata=metadata,
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
            runtime_details=runner.runtime_details(),
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
            if summary.final_state_kind != request.method:
                raise BackendExecutionError(
                    "pulse execution produced a final state inconsistent with "
                    f"method={request.method!r}"
                )
            available.add(request.method)
            if request.method == "statevector":
                statevector = final_state
            else:
                density_matrix = final_state
        if request.counts and not request.final_state:
            n_clbits = prepared.classical_allocation.n_clbits
            if any(
                classical_index not in prepared.facts.written_clbits
                for classical_index in range(n_clbits)
            ):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    stacklevel=4,
                )
        metadata = self._result_metadata(request, simulation, shots)
        if request.final_state:
            metadata["state_axes"] = _describe_state_axes(
                prepared.engine_allocation.device_operands,
                prepared.resource_layout,
            )
        metadata["runtime_details"] = dict(summary.runtime_details)
        return Result(
            counts=counts,
            statevector=statevector,
            density_matrix=density_matrix,
            available=frozenset(available),
            classical_dims=prepared.classical_allocation.classical_dims,
            metadata=metadata,
        )

    @final
    def _result_metadata(
        self,
        request: _PulseResultRequest,
        simulation: _EmulatorConfig,
        shots: int,
    ) -> dict[str, Any]:
        return {
            "backend_name": self._backend_name(),
            "method": request.method,
            "runtime": self._runtime_name,
            "runtime_details": {
                "solver": "none",
                "solver_options": {},
            },
            "shots": shots,
            "simulation_config": asdict(simulation),
            "result_config": {
                "counts": request.counts,
                "final_state": request.final_state,
            },
        }

    @final
    def _backend_name(self) -> str:
        """Return the public backend name used in errors and metadata."""
        return type(self).__name__

    def _validate_source_program(self, program: Program) -> None:
        """Validate only family-specific source-language restrictions."""
        del program

    @abstractmethod
    def _noise_model_rejection_reasons(
        self, noise_model: NoiseModel
    ) -> tuple[str, ...]:
        """Return family rejection reasons for an already typed noise model."""
        raise NotImplementedError

    @abstractmethod
    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        simulation: _EmulatorConfig,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        """Create a runner from already-bound execution data."""
        del prepared, simulation, execution_mode, retain_final_state
        raise NotImplementedError

    @final
    def validate_noise_model(self, noise_model: NoiseModel) -> None:
        """Raise if this emulator cannot use a noise model.

        This validation does not inspect a particular program or resource
        layout. Program references and physical selectors are checked when you
        call ``run()``.

        Args:
            noise_model: Noise model to validate without executing a program.

        Raises:
            BackendValidationError: If ``noise_model`` is not a
                ``fatqat.NoiseModel`` or contains unsupported declarations.
                The message lists every problem found.
        """
        if not isinstance(noise_model, NoiseModel):
            raise BackendValidationError("noise_model must be a NoiseModel")
        rejection_reasons = self._noise_model_rejection_reasons(noise_model)
        if rejection_reasons:
            raise BackendValidationError("; ".join(rejection_reasons))
