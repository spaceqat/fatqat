"""Three-level neutral-atom pulse emulator."""

from __future__ import annotations

from typing import Any

from ...atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...noise import (
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
)
from .._core.backend import _PulseBackend
from .._core.lindblad import _classify_lindblad_noise
from .._core.outcome import ExecutionMode
from .._core.planning import PulsePlanFacts, _PreparedPulseProgram
from .._core.pulse import PulseImplementationMap
from .calibration import default_atom_3level_calibration
from .model import Atom3LevelModel
from .realization import default_atom_3level_gate_implementation_map
from .target import _Atom3LevelTarget


class Atom3LevelEmulator(_PulseBackend):
    """Calibrated three-level neutral-atom emulator in ``|0>, |1>, |r>``.

    Program resources remain dimension two, while physical evolution and
    returned density matrices retain all three local levels. The built-in
    gate map realizes ``RX``, ``RY``, ``RZ``, and ``CZ``. A supplied gate map
    replaces that default, and a supplied Lindblad map enables the registered
    physical channel descriptors. No physical channels are registered by
    default; binary classical readout confusion remains supported.
    """

    _coherent_execution_mode: ExecutionMode = "density_matrix"

    def __init__(
        self,
        model: Atom3LevelModel,
        *,
        arrangement: AtomArrangement,
        noise: NoiseModel | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
    ) -> None:
        if not isinstance(model, Atom3LevelModel):
            raise BackendValidationError("model must be an Atom3LevelModel")
        if not isinstance(arrangement, AtomArrangement):
            raise BackendValidationError("arrangement must be an AtomArrangement")

        effective_gate_map = (
            default_atom_3level_gate_implementation_map(
                model=model,
                calibration=default_atom_3level_calibration(),
            )
            if gate_implementation_map is None
            else gate_implementation_map
        )
        effective_lindblad_map = (
            LindbladImplementationMap()
            if lindblad_implementation_map is None
            else lindblad_implementation_map
        )
        self._arrangement = arrangement
        super().__init__(
            model,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=effective_lindblad_map,
        )
        self._require_captured_noise_support()
        self._set_target(_Atom3LevelTarget(model, arrangement))

    @property
    def arrangement(self) -> AtomArrangement:
        """Return the arrangement bound to this emulator."""

        return self._arrangement

    def _classify_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        return _classify_lindblad_noise(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=self.model.local_dimension,
            backend_name=type(self).__name__,
            supports_readout_confusion=True,
            readout_confusion_shape=(2, 2),
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
                "Atom3LevelEmulator supports only density-matrix execution"
            )
        from .qutip_adapter import _Atom3LevelQutipAdapter

        return _Atom3LevelQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            retain_final_state=retain_final_state,
        )


__all__ = ["Atom3LevelEmulator"]
