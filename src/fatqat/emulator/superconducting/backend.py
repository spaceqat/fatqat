"""Superconducting transmon pulse emulator."""

from __future__ import annotations

from typing import Any

from ...errors import BackendValidationError
from ...noise import (
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    default_lindblad_implementation_map,
)
from .._core.backend import _PulseBackend
from .._core.lindblad import _classify_lindblad_noise
from .._core.outcome import ExecutionMode
from .._core.planning import PulsePlanFacts, _PreparedPulseProgram
from .._core.pulse import PulseImplementationMap
from .calibration import default_transmon_calibration
from .model import TransmonModel
from .realization import default_transmon_gate_implementation_map
from .target import _TransmonTarget


class TransmonEmulator(_PulseBackend):
    """Simulate calibrated controls on a fixed three-level transmon model."""

    _coherent_execution_mode: ExecutionMode = "density_matrix"

    def __init__(
        self,
        model: TransmonModel,
        *,
        noise: NoiseModel | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
    ) -> None:
        if not isinstance(model, TransmonModel):
            raise BackendValidationError("model must be a TransmonModel")
        effective_gate_map = (
            default_transmon_gate_implementation_map(
                model=model,
                calibration=default_transmon_calibration(),
            )
            if gate_implementation_map is None
            else gate_implementation_map
        )
        effective_lindblad_map = (
            default_lindblad_implementation_map()
            if lindblad_implementation_map is None
            else lindblad_implementation_map
        )
        super().__init__(
            model,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=effective_lindblad_map,
        )
        self._set_target(_TransmonTarget(model))

    def _classify_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        return _classify_lindblad_noise(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=self.model.local_dimension,
            backend_name=type(self).__name__,
            supports_readout_confusion=True,
        )

    def _resolve_execution_mode(self, facts: PulsePlanFacts) -> ExecutionMode:
        del facts
        return "density_matrix"

    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        if execution_mode != "density_matrix":
            raise BackendValidationError(
                "TransmonEmulator supports only density-matrix execution"
            )
        from .qutip_adapter import _TransmonQutipAdapter

        return _TransmonQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            retain_final_state=retain_final_state,
        )


__all__ = ["TransmonEmulator"]
