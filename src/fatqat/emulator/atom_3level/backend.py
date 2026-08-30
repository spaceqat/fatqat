"""Three-level neutral-atom pulse emulator."""

from __future__ import annotations

from typing import Any

from ..atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...noise import (
    NoiseModel,
)
from ...noise.lindblad import LindbladImplementationMap
from .._core.backend import _PulseBackend
from .._core.lindblad import _lindblad_noise_rejection_reasons
from .._core.outcome import ExecutionMode
from .._core.planning import _PreparedPulseProgram
from .._core.pulse import PulseImplementationMap
from .calibration import default_atom_3level_calibration
from .model import Atom3LevelModel
from .realization import default_atom_3level_gate_implementation_map
from .target import _Atom3LevelTarget


class Atom3LevelEmulator(_PulseBackend):
    """Run calibrated gates and local controls in ``|0>, |1>, |r>``.

    Programs use dimension-two resources, while evolution and returned density
    matrices retain all three physical levels. Every run starts with each atom
    in ``|0>``.

    Args:
        model: Three-level atom model created with
            ``Atom3LevelModel.from_document``.
        arrangement: Fixed site coordinates for the program. The program must
            contain one dimension-two resource per site.
        noise: Noise applied by this emulator. The default is no noise.
        gate_implementation_map: Gate-to-pulse rules. ``None`` uses the
            built-in ``RX``, ``RY``, ``RZ``, and ``CZ`` rules.
    Raises:
        BackendValidationError: If an argument has the wrong type or ``noise``
            contains a declaration unsupported by the selected rules.

    Examples:
        >>> import fatqat as fq
        >>> model = fq.emulator.Atom3LevelModel.from_document(
        ...     fq.emulator.load_model_document("atom3level.reference")
        ... )
        >>> arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
        >>> backend = fq.emulator.Atom3LevelEmulator(
        ...     model, arrangement=arrangement
        ... )
        >>> backend.arrangement.num_sites
        2
    """

    def __init__(
        self,
        model: Atom3LevelModel,
        *,
        arrangement: AtomArrangement,
        method: str = "statevector",
        noise: NoiseModel | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
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
        self._arrangement = arrangement
        super().__init__(
            model,
            method=method,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=LindbladImplementationMap(),
        )
        self.validate_noise_model(self._noise_model)
        self._set_target(_Atom3LevelTarget(model, arrangement))

    @property
    def arrangement(self) -> AtomArrangement:
        """Return the arrangement bound to this emulator."""

        return self._arrangement

    def _noise_model_rejection_reasons(
        self, noise_model: NoiseModel
    ) -> tuple[str, ...]:
        return _lindblad_noise_rejection_reasons(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=self.model.local_dimension,
            backend_name=type(self).__name__,
            supports_readout_confusion=True,
            readout_confusion_shape=(2, 2),
        )

    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        if execution_mode not in ("statevector", "density_matrix"):
            raise BackendValidationError(
                "Atom3LevelEmulator does not support continuous trajectories"
            )
        from .qutip_adapter import _Atom3LevelQutipAdapter

        return _Atom3LevelQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            execution_mode=execution_mode,
            retain_final_state=retain_final_state,
        )


__all__ = ["Atom3LevelEmulator"]
