"""Superconducting pulse backend with private full-qutrit execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

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
from .superconducting import CalibrationSpec, PhysicsModel
from .superconducting_realization import (
    default_superconducting_pulse_implementation_map,
)


class PulseBackend:
    """SC pulse backend over an immutable model and separate calibration."""

    def __init__(
        self,
        model: PhysicsModel,
        calibration: CalibrationSpec,
        *,
        noise: NoiseModel | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
        pulse_implementation_map: PulseImplementationMap | None = None,
    ) -> None:
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
        return _EngineIndexAllocation.from_program(program)

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
        """Validate/lower one pulse program and return an eager terminal job."""
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
        except Exception:  # execution failures belong on the eager Job
            return Job.failed(BackendExecutionError("Pulse backend execution failed"))

    def _validate(
        self, config: PulseResultConfig, shots: int, facts: PulsePlanFacts
    ) -> _DensityMatrixResultRequest:
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
        from .qutip_adapter import SCQutipAdapter

        runner = SCQutipAdapter(
            self.model,
            engine_index_to_model_ordinal=engine_index_to_model_ordinal,
            always_on_noise=always_on_noise,
        )
        outcomes = PulseEngine(runner, placement_mode=simulation.placement_mode).run(
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
        """Report support by descriptor parameterization and activation scope."""
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
