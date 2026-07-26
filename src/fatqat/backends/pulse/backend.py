"""Superconducting pulse backend with private full-qutrit execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ..._engine_index_allocation import _EngineIndexAllocation
from ...backends.backend_utils import _LoweringContext, _normalize_config
from ...errors import BackendExecutionError, BackendValidationError
from ...job import Job
from ...noise import NoiseModel, NoiseSupportReport, ThermalRelaxation
from ...program import Program
from ...resource_layout import ResourceLayout
from ...result import Result, counts_dict_from_arrays, reduce_to_counts
from .engine import PulseEngine
from .engine_contract import (
    PulseResultConfig,
    PulseResultRequest,
    PulseSimulationConfig,
)
from .planning import PulsePlanFacts, PulsePlanStep, lower_program
from .superconducting import CalibrationSpec, PhysicsModel


class PulseBackend:
    """SC pulse backend over an immutable model and separate calibration."""

    def __init__(
        self,
        model: PhysicsModel,
        calibration: CalibrationSpec,
        *,
        noise: NoiseModel | None = None,
    ) -> None:
        if calibration.key != model.key:
            raise BackendValidationError("calibration does not match the pulse model")
        self.model = model
        self.calibration = calibration
        self.noise = noise or NoiseModel()

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
        """Bind program declaration order to the snapshot's ordered subsystem ids."""
        labels: dict[Any, str] = {}
        refs = [
            register[index]
            for register in program.qreg
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
        return lower_program(
            program,
            model=self.model,
            calibration=self.calibration,
            noise_model=self.noise,
            resource_layout=context.resource_layout,
            engine_index_allocation=context.engine_index_allocation,
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
        self.noise.validate_for(program, resource_layout)
        report = self.validate_noise(self.noise)
        if not report.supported:
            raise BackendValidationError("; ".join(report.warnings))
        context = _LoweringContext(
            resource_layout=resource_layout,
            engine_index_allocation=allocation,
        )
        plan, facts = self._lower_program(program, context=context)
        request = self._validate(result, shots, facts)
        engine_to_model = self._engine_to_model(program, resource_layout, allocation)
        continuous_noise = self._continuous_noise(program, resource_layout)
        try:
            return Job.done(
                self._execute(
                    plan,
                    request,
                    simulation,
                    result,
                    shots,
                    allocation,
                    engine_to_model,
                    continuous_noise,
                )
            )
        except Exception:  # execution failures belong on the eager Job
            return Job.failed(BackendExecutionError("Pulse backend execution failed"))

    def _validate(
        self, config: PulseResultConfig, shots: int, facts: PulsePlanFacts
    ) -> PulseResultRequest:
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
        return PulseResultRequest(counts=counts, density_matrix=density_matrix)

    def _execute(
        self,
        plan: list[PulsePlanStep],
        request: PulseResultRequest,
        simulation: PulseSimulationConfig,
        result_config: PulseResultConfig,
        shots: int,
        allocation: _EngineIndexAllocation,
        engine_to_model: tuple[int, ...],
        continuous_noise: tuple[tuple[Any, ...], ...],
    ) -> Result:
        from .qutip_adapter import SCQutipAdapter

        runner = SCQutipAdapter(
            self.model,
            engine_to_model=engine_to_model,
            continuous_noise=continuous_noise,
        )
        outcomes = PulseEngine(
            runner, placement_mode=simulation.placement_mode
        ).execute(
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

    def _engine_to_model(
        self,
        program: Program,
        resource_layout: ResourceLayout,
        allocation: _EngineIndexAllocation,
    ) -> tuple[int, ...]:
        ordinals = [0] * allocation.n_subsystems
        for register in program.qreg:
            for index in range(register.size):
                ref = register[index]
                label = resource_layout.device_label(ref)
                ordinals[allocation.subsystem_index(ref)] = self.model.bind_resource(
                    self.model.resource(label)
                )
        return tuple(ordinals)

    def _continuous_noise(
        self, program: Program, resource_layout: ResourceLayout
    ) -> tuple[tuple[Any, ...], ...]:
        refs_by_label = {
            resource_layout.device_label(register[index]): register[index]
            for register in program.qreg
            for index in range(register.size)
        }
        return tuple(
            self.noise.continuous_noise_for(
                refs_by_label.get(subsystem_id), subsystem_id
            )
            for subsystem_id in self.model.subsystem_ids
        )

    def validate_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report accepted T1/T2/readout and rejected pulse-noise sources."""
        accepted = ["readout_error"] if noise_model.has_readout_error() else []
        rejected = []
        warnings = []
        for channel_type in sorted(
            noise_model.channel_types(), key=lambda source: source.__name__
        ):
            rejected.append(channel_type.__name__)
            warnings.append(
                f"{channel_type.__name__} gate-keyed channel noise is not supported "
                "by the pulse backend"
            )
        for source_type in sorted(
            noise_model.continuous_noise_types(), key=lambda source: source.__name__
        ):
            if source_type is ThermalRelaxation:
                accepted.append(source_type.__name__)
            else:
                rejected.append(source_type.__name__)
                warnings.append(
                    f"{source_type.__name__} is not supported by the pulse backend"
                )
        if noise_model.qubit_noise:
            rejected.append("qubit_noise")
            warnings.append(
                "qubit_noise is a legacy placeholder; use add_continuous_noise()"
            )
        return NoiseSupportReport(
            supported=not rejected,
            accepted_sources=tuple(accepted),
            rejected_sources=tuple(rejected),
            warnings=tuple(warnings),
        )
