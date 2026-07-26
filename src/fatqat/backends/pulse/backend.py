"""Planning-only superconducting pulse backend shell.

The future engine stages consume this backend's unplaced plan.  Until then,
successful validation/lowering returns an eager failed job rather than
pretending that a matrix-family simulator executed the pulse program.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..._engine_index_allocation import _EngineIndexAllocation
from ...backends.backend_utils import _normalize_config
from ...errors import BackendValidationError
from ...job import Job
from ...noise import NoiseModel, NoiseSupportReport
from ...program import Program
from ...result import Result
from ...resource_layout import ResourceLayout
from .engine_contract import (
    PulseResultConfig,
    PulseResultRequest,
    PulseSimulationConfig,
)
from .planning import PulsePlanFacts, PulsePlanStep, lower_program
from .execution import execute_with_boundaries
from .qutip_adapter import SCQutipAdapter
from .superconducting import CalibrationSpec, PhysicsModel


class PulseBackend:
    """SC pulse lowering shell; execution becomes available in later tasks."""

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
        resource_layout: ResourceLayout | None = None,
        engine_index_allocation: _EngineIndexAllocation | None = None,
    ) -> tuple[list[PulsePlanStep], PulsePlanFacts]:
        if resource_layout is None:
            resource_layout = self._resolve_resource_layout(program)
        if engine_index_allocation is None:
            engine_index_allocation = self._allocate_engine_indices(program)
        return lower_program(
            program,
            model=self.model,
            calibration=self.calibration,
            noise_model=self.noise,
            resource_layout=resource_layout,
            engine_index_allocation=engine_index_allocation,
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
        plan, facts = self._lower_program(
            program,
            resource_layout=resource_layout,
            engine_index_allocation=allocation,
        )
        self._validate(result, shots, facts)
        try:
            return Job.done(self._execute_continuous(plan, simulation, result, shots))
        except Exception as exc:  # execution failures belong on the eager Job
            return Job.failed(exc)

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

    def _execute_continuous(
        self,
        plan: list[PulsePlanStep],
        simulation: PulseSimulationConfig,
        result: PulseResultConfig,
        shots: int,
    ) -> Result:
        """Run an ideal boundary-free continuous region in private qutrit space."""
        request = self._validate(result, shots, PulsePlanFacts(False, False, False))
        adapter = SCQutipAdapter(self.model)

        def unsupported_boundary(_step: object, _time_ns: float) -> None:
            raise RuntimeError(
                "physical measurement and reset execution is unavailable until dynamic replay is implemented"
            )

        execute_with_boundaries(plan, adapter.evolve, unsupported_boundary)
        density_matrix = adapter.density_matrix()
        available = {"density_matrix"} if request.density_matrix else set()
        return Result(
            density_matrix=density_matrix if request.density_matrix else None,
            available=frozenset(available),
            metadata={
                "backend_name": type(self).__name__,
                "shots": shots,
                "simulation_config": asdict(simulation),
                "result_config": asdict(result),
            },
        )

    def validate_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        """Report v0.1 planning support before continuous-noise work arrives."""
        accepted = ("readout_error",) if noise_model.has_readout_error() else ()
        rejected = []
        warnings = []
        if noise_model.channel_types():
            rejected.append("gate_channel_noise")
            warnings.append(
                "gate-keyed channel noise is not supported by the pulse backend"
            )
        if noise_model.qubit_noise:
            rejected.append("qubit_noise")
            warnings.append("continuous pulse noise is introduced in Task 7")
        return NoiseSupportReport(
            supported=not rejected,
            accepted_sources=accepted,
            rejected_sources=tuple(rejected),
            warnings=tuple(warnings),
        )
