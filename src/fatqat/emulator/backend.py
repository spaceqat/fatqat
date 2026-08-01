"""Superconducting pulse backend with private full-qutrit execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal

import numpy as np

from .._engine_index_allocation import _EngineIndexAllocation
from ..backends.backend_utils import _LoweringContext, _normalize_config
from ..backends.engine_contract import _DensityMatrixResultRequest
from ..backends.steps import MeasurementStep
from ..backends.view_normalization import ProgramInstruction, _break_grouped_operations
from ..errors import BackendExecutionError, BackendValidationError
from ..job import Job
from ..noise import (
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    default_lindblad_implementation_map,
)
from ..noise.lindblad import resolve_lindblad_operators
from ..operations import BarrierGate, Measurement, ResetGate
from ..program import AppliedOperation, Program
from ..resource_layout import ResourceLayout
from ..result import Result, counts_dict_from_arrays, reduce_to_counts
from . import planning
from .engine import PulseEngine
from .engine_contract import PulseResultConfig, PulseSimulationConfig
from .planning import PulsePlanFacts, PulsePlanStep
from .lindblad import ResolvedLindbladTerm, bind_lindblad_operators
from .pulse import PulseImplementationMap
from .scheduling import _validate_schedule_mode
from .superconducting import CalibrationSpec, SCTransmonModel
from .superconducting_realization import (
    default_superconducting_pulse_implementation_map,
)


class PulseBackend:
    """Simulate calibrated controls on a fixed three-level transmon model.

    ``PulseBackend`` is the public entry point for the superconducting pulse
    emulator. A backend is constructed from an immutable physics model and a
    separately loaded calibration that is identity-bound to that exact model.
    Program qubits bind to model subsystems in declaration order; every model
    subsystem remains in the simulated Hilbert space, including subsystems a
    program does not address.

    The built-in implementation map accepts ``RX``, ``RY``, virtual ``RZ``,
    ``iSwap``, and oriented ``CZ`` operations on declared coupling edges.
    Measurement collapses the physical qutrit and reports levels ``0, 1, 2``
    as classical digits ``0, 1, 1``. Reset prepares the selected qutrit in
    its physical ground state.

    ``run()`` performs open-system evolution and returns counts and/or the
    full physical density matrix. ``propagator()`` is the coherent-analysis
    path and returns the full-model operator when the program contains no
    boundary operations, classical conditions, or elapsed dissipative
    evolution. Neither method exposes QuTiP objects.

    A :class:`~fatqat.NoiseModel` may contain always-on rate-based damping,
    operation-scoped damping in probability or rate mode, and readout
    confusion. The optional Lindblad and pulse implementation maps are copied
    at construction, while the supplied noise model is retained by reference,
    matching :class:`~fatqat.backends.SimulatorBackend`'s noise ownership.

    The built-in CZ realization derives its nominal virtual frame correction
    from the detuning waveform itself. This first-version model correction is
    intentionally not a hardware phase calibration; device-specific phase
    calibration can further improve the realized gate quality in the future.

    A backend instance has no mutable solver state between calls. Individual
    calls execute eagerly and serially in v0.1.
    """

    def __init__(
        self,
        model: SCTransmonModel,
        calibration: CalibrationSpec,
        *,
        noise: NoiseModel | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
        pulse_implementation_map: PulseImplementationMap | None = None,
    ) -> None:
        """Create a pulse backend for one model/calibration pair.

        Args:
            model: Physics model returned by
                :func:`~fatqat.backends.load_physics_model`.
            calibration: Calibration returned by
                :func:`~fatqat.backends.load_calibration_spec` for ``model``.
            noise: Optional noise model. ``None`` creates an empty model. A
                supplied model is retained by reference so later registrations
                affect subsequent runs.
            lindblad_implementation_map: Optional mapping from supported
                channel descriptors to local collapse operators. ``None`` uses
                the default map. A supplied map is copied immediately.
            pulse_implementation_map: Optional operation-to-pulse realization
                map. ``None`` uses
                :func:`~fatqat.backends.default_superconducting_pulse_implementation_map`.
                A supplied map is copied immediately.

        Raises:
            BackendValidationError: If ``calibration`` belongs to a different
                model snapshot.
        """
        if calibration.key != model.key:
            raise BackendValidationError("calibration does not match the pulse model")
        self.model = model
        self.calibration = calibration
        self._noise_model = noise or NoiseModel()
        self._lindblad_implementation_map = (
            default_lindblad_implementation_map()
            if lindblad_implementation_map is None
            else lindblad_implementation_map.copy()
        )
        self._pulse_implementation_map = (
            default_superconducting_pulse_implementation_map()
            if pulse_implementation_map is None
            else pulse_implementation_map.copy()
        )

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
        """Bind program declaration order to the snapshot's ordered subsystem ids."""
        labels: dict[Any, str] = {}
        refs = [
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        ]
        if len(refs) > len(self.model.subsystems):
            raise BackendValidationError(
                f"program requires {len(refs)} subsystems but model has {len(self.model.subsystems)}"
            )
        for ref in refs:
            if ref.register.dim != 2:
                raise BackendValidationError(
                    "PulseBackend embeds only dimension-two program subsystems into qutrits"
                )
        for ordinal, ref in enumerate(refs):
            labels[ref] = self.model.subsystem_ids[ordinal]
        return ResourceLayout(labels)

    @staticmethod
    def _allocate_engine_indices(program: Program) -> _EngineIndexAllocation:
        """Build the private flat subsystem/classical allocation for one run."""
        return _EngineIndexAllocation.from_program(program)

    def _prepare_program(self, program: Program) -> tuple[
        list[PulsePlanStep],
        PulsePlanFacts,
        ResourceLayout,
        _EngineIndexAllocation,
    ]:
        """Validate and lower one program for execution or propagation."""
        resource_layout = self._resolve_resource_layout(program)
        allocation = self._allocate_engine_indices(program)
        self._noise_model.validate_for(program, resource_layout)
        report = self.validate_noise(self._noise_model)
        if not report.supported:
            raise BackendValidationError("; ".join(report.warnings))
        context = _LoweringContext(
            resource_layout=resource_layout,
            engine_index_allocation=allocation,
        )
        plan, facts = self._lower_program(program, context=context)
        return plan, facts, resource_layout, allocation

    def _lower_program(
        self,
        program: Program,
        *,
        context: _LoweringContext | None = None,
    ) -> tuple[list[PulsePlanStep], PulsePlanFacts]:
        """Prepare and lower one program using the backend's resource policy.

        ``context`` lets a caller that already resolved this run's
        `ResourceLayout` and `_EngineIndexAllocation` (see ``run()``) thread
        both through unchanged, so lowering never re-resolves either. When
        omitted (standalone use, e.g. in tests), both are resolved once here
        - always together, never as two independently-defaulted halves.
        """
        if context is None:
            context = _LoweringContext(
                resource_layout=self._resolve_resource_layout(program),
                engine_index_allocation=self._allocate_engine_indices(program),
            )
        operations = _break_grouped_operations(program.operations)
        return self._lower(operations, context)

    def _lower(
        self,
        operations: Sequence[ProgramInstruction],
        context: _LoweringContext,
    ) -> tuple[list[PulsePlanStep], PulsePlanFacts]:
        """Lower scalar instructions into an ordered, unplaced pulse plan."""
        resource_layout = context.resource_layout
        engine_index_allocation = context.engine_index_allocation
        plan: list[PulsePlanStep] = []
        for step in operations:
            if isinstance(step, Measurement):
                plan.append(
                    planning._lower_measurement(
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
                    plan.append(planning._lower_reset(step, engine_index_allocation))
                else:
                    plan.append(
                        planning._lower_gate(
                            step,
                            resource_layout,
                            engine_index_allocation,
                            self.model,
                            self.calibration,
                            self._pulse_implementation_map,
                            self._noise_model,
                            self._lindblad_implementation_map,
                        )
                    )
        return plan, PulsePlanFacts(
            has_measurement=any(isinstance(step, MeasurementStep) for step in plan)
        )

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        simulation_config: dict[str, Any] | None = None,
        result_config: dict[str, Any] | None = None,
    ) -> Job:
        """Validate, execute, and package one pulse-program run.

        ``simulation_config`` accepts ``seed``, ``parallel_mode``,
        ``max_workers``, and ``schedule_mode``. Pulse execution is serial in
        v0.1, so ``parallel_mode`` may be ``"auto"`` or ``"serial"`` and
        ``max_workers`` may be ``None`` or ``1``. ``schedule_mode`` is
        ``"ASAP"`` by default and may be ``"ALAP"``; both are lightweight
        placement policies over dependencies and claimed physical resources,
        not compiler-produced hardware schedules.

        ``result_config`` accepts ``counts`` and ``final_state``. When omitted,
        counts default on for programs containing measurement and the final
        state defaults on for programs without measurement. ``final_state``
        is a full physical density matrix with shape ``(3**m, 3**m)`` for the
        model's ``m`` transmons. A measured final state is one sampled
        posterior and therefore requires ``shots == 1``.

        Args:
            program: Program to bind, lower, and execute.
            shots: Number of repetitions used when counts are requested.
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
            BackendValidationError: If the model cannot bind the program,
                noise/configuration is unsupported, or requested results are
                incompatible with ``shots`` and measurement.
            UnsupportedOperationError: If no pulse implementation exists for
                an operation family or its ordered device operands.
            PulseImplementationError: If a selected custom pulse rule fails
                unexpectedly or returns the wrong value type.
        """
        simulation = _normalize_config(
            simulation_config,
            PulseSimulationConfig,
            "simulation_config",
            backend_name=type(self).__name__,
        )
        result = _normalize_config(
            result_config,
            PulseResultConfig,
            "result_config",
            backend_name=type(self).__name__,
        )
        plan, facts, resource_layout, allocation = self._prepare_program(program)
        request = self._validate(result, shots, facts)
        engine_index_to_model_ordinal = self._engine_index_to_model_ordinal(
            program, resource_layout, allocation
        )
        always_on_noise = self._always_on_noise(program, resource_layout)
        try:
            return Job.done(
                self._execute(
                    plan,
                    request,
                    simulation,
                    result,
                    shots,
                    allocation,
                    engine_index_to_model_ordinal,
                    always_on_noise,
                )
            )
        except Exception as exc:  # execution failures belong on the eager Job
            # The public message stays stable and free of solver internals,
            # but the original exception is chained so a developer (and a
            # traceback) can still see what actually failed. Assigning
            # `__cause__` rather than raising keeps this an eager failed Job.
            failure = BackendExecutionError("Pulse backend execution failed")
            failure.__cause__ = exc
            return Job.failed(failure)

    def propagator(
        self,
        program: Program,
        *,
        apply_final_frame: bool = True,
        schedule_mode: Literal["ASAP", "ALAP"] = "ASAP",
    ) -> np.ndarray:
        """Return the coherent full-model propagator for ``program``.

        Intermediate virtual-frame updates always rotate later phase-sensitive
        controls. By default the remaining terminal frame transformation is
        also composed onto the returned propagator; set
        ``apply_final_frame=False`` to inspect Hamiltonian-generated evolution
        before that final basis transformation.

        The result is a complex NumPy array of shape ``(3**m, 3**m)`` for the
        model's ``m`` transmons, expressed in the model's near-resonant
        rotating frame. Its virtual-Z representation can differ from a
        conventional qubit RZ matrix by a global phase. Measurement, reset,
        and classical conditions are rejected. Bound collapse terms are
        rejected when the plan contains nonzero elapsed evolution; rate-based
        noise has no effect on a frame-only plan because no time elapses.

        Args:
            program: Coherent program to lower, schedule, and propagate.
            apply_final_frame: Whether to compose the terminal virtual-frame
                transformation. Intermediate frame updates are always honored.
            schedule_mode: Lightweight pulse placement policy, ``"ASAP"`` or
                ``"ALAP"``.

        Returns:
            Full-model coherent propagator as a NumPy array.
        """
        if type(apply_final_frame) is not bool:
            raise BackendValidationError("apply_final_frame must be a bool")
        schedule_mode = _validate_schedule_mode(schedule_mode)
        plan, _facts, resource_layout, allocation = self._prepare_program(program)

        if not plan:
            dimension = self.model.physical_dimension ** len(self.model.subsystems)
            return np.eye(dimension, dtype=complex)

        from .qutip_adapter import SCQutipAdapter

        runner = SCQutipAdapter(
            self.model,
            engine_index_to_model_ordinal=self._engine_index_to_model_ordinal(
                program, resource_layout, allocation
            ),
            always_on_noise=self._always_on_noise(program, resource_layout),
        )
        engine = PulseEngine(runner, schedule_mode=schedule_mode)
        try:
            return np.asarray(
                engine.propagator(plan, apply_final_frame=apply_final_frame).full(),
                dtype=complex,
            )
        except BackendValidationError:
            raise
        except Exception as exc:
            raise BackendExecutionError("Pulse propagator construction failed") from exc

    def _validate(
        self, config: PulseResultConfig, shots: int, facts: PulsePlanFacts
    ) -> _DensityMatrixResultRequest:
        """Resolve default output requests and validate their shot constraints."""
        counts = config.counts if config.counts is not None else facts.has_measurement
        density_matrix = (
            config.final_state
            if config.final_state is not None
            else not facts.has_measurement
        )
        if (counts or (config.final_state is True and facts.has_measurement)) and type(
            shots
        ) is not int:
            raise BackendValidationError(
                "shots must be an int when requested results depend on it"
            )
        if counts and shots <= 0:
            raise BackendValidationError(f"counts require shots > 0, got shots={shots}")
        if config.final_state is True and facts.has_measurement and shots != 1:
            raise BackendValidationError(
                "density_matrix with physical measurement sampling is only supported for shots == 1"
            )
        return _DensityMatrixResultRequest(counts=counts, density_matrix=density_matrix)

    def _execute(
        self,
        plan: list[PulsePlanStep],
        request: _DensityMatrixResultRequest,
        simulation: PulseSimulationConfig,
        result_config: PulseResultConfig,
        shots: int,
        allocation: _EngineIndexAllocation,
        engine_index_to_model_ordinal: tuple[int, ...],
        always_on_noise: tuple[ResolvedLindbladTerm, ...],
    ) -> Result:
        """Execute a validated plan and convert private shot payloads to Result."""
        from .qutip_adapter import SCQutipAdapter

        runner = SCQutipAdapter(
            self.model,
            engine_index_to_model_ordinal=engine_index_to_model_ordinal,
            always_on_noise=always_on_noise,
        )
        outcomes = PulseEngine(runner, schedule_mode=simulation.schedule_mode).run(
            plan,
            shots=shots if request.counts else 1,
            n_clbits=allocation.n_clbits,
            rng=np.random.default_rng(simulation.seed),
        )
        density_matrix = outcomes[-1].density_matrix if outcomes else None
        counts = None
        available = set()
        if request.counts:
            keys, values = reduce_to_counts(
                [outcome.classical_digits for outcome in outcomes]
            )
            counts = counts_dict_from_arrays(keys, values)
            available.add("counts")
        if request.density_matrix:
            available.add("density_matrix")
        return Result(
            counts=counts,
            density_matrix=density_matrix if request.density_matrix else None,
            available=frozenset(available),
            classical_dims=allocation.classical_dims,
            metadata={
                "backend_name": type(self).__name__,
                "shots": shots,
                "simulation_config": asdict(simulation),
                "result_config": asdict(result_config),
                "solver": runner.solver_metadata(),
            },
        )

    def _engine_index_to_model_ordinal(
        self,
        program: Program,
        resource_layout: ResourceLayout,
        allocation: _EngineIndexAllocation,
    ) -> tuple[int, ...]:
        """Build the engine-index to physical-model ordinal translation."""
        ordinals = [0] * allocation.n_subsystems
        for register in program.quantum_registers:
            for index in range(register.size):
                ref = register[index]
                label = resource_layout.device_label(ref)
                ordinals[allocation.subsystem_index(ref)] = self.model.bind_resource(
                    self.model.resource(label)
                )
        return tuple(ordinals)

    def _always_on_noise(
        self, program: Program, resource_layout: ResourceLayout
    ) -> tuple[ResolvedLindbladTerm, ...]:
        """Resolve all always-on descriptors into pulse Lindblad terms."""
        refs_by_label = {
            resource_layout.device_label(register[index]): register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        }
        bindings: list[ResolvedLindbladTerm] = []
        for ordinal, subsystem_id in enumerate(self.model.subsystem_ids):
            for channel in self._noise_model.always_on_channels_for(
                refs_by_label.get(subsystem_id), subsystem_id
            ):
                bindings.extend(
                    bind_lindblad_operators(
                        resolve_lindblad_operators(
                            channel,
                            implementation_map=self._lindblad_implementation_map,
                            physical_dimension=self.model.physical_dimension,
                            duration=None,
                        ),
                        model_ordinals=(ordinal,),
                    )
                )
        return tuple(bindings)

    def validate_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report whether this backend can realize a noise model.

        Support is instance-sensitive. Damping descriptors are reported by
        parameterization (``p`` or ``rate``) and activation scope because an
        always-on descriptor requires a rate, while operation-scoped damping
        may use either form. Readout confusion is reported separately.

        This capability check does not validate selectors against a particular
        program; :meth:`run` and :meth:`propagator` perform that separate
        :meth:`fatqat.NoiseModel.validate_for` step after resolving the
        program's resource layout.

        Args:
            noise_model: Noise registrations to classify without executing a
                program.

        Returns:
            A :class:`~fatqat.noise.NoiseSupportReport` listing accepted and
            rejected source descriptions plus explanatory warnings.
        """
        accepted = ["readout_error"] if noise_model.has_readout_error() else []
        rejected = []
        warnings = []
        seen: set[str] = set()
        for channel, operation in noise_model.channel_registrations():
            channel_type = type(channel)
            always_on = operation is None
            qualifiers: list[str] = []
            if hasattr(channel, "rate"):
                qualifiers.append("rate" if channel.rate is not None else "p")
            if always_on:
                qualifiers.append("always-on")
            label = channel_type.__name__
            if qualifiers:
                label += f"({', '.join(qualifiers)})"
            if label in seen:
                continue
            seen.add(label)
            supported = (
                channel_type in self._lindblad_implementation_map.supported_channels()
            )
            if always_on and hasattr(channel, "rate") and channel.rate is None:
                supported = False
            if supported:
                accepted.append(label)
            else:
                rejected.append(label)
                if always_on and hasattr(channel, "rate") and channel.rate is None:
                    warnings.append(
                        f"{label} is not supported: always-on damping requires "
                        "rate mode"
                    )
                else:
                    warnings.append(
                        f"{label} has no pulse channel implementation on this backend"
                    )
        return NoiseSupportReport(
            supported=not rejected,
            accepted_sources=tuple(accepted),
            rejected_sources=tuple(rejected),
            warnings=tuple(warnings),
        )
