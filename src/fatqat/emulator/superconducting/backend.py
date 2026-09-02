"""Superconducting transmon pulse emulator."""

from __future__ import annotations

from typing import Any

from ...errors import BackendValidationError
from ...noise import (
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    TransitionRelaxation,
)
from ...noise.lindblad import (
    LindbladImplementationMap,
    depolarizing_lindblad_rule,
    phase_damping_lindblad_rule,
    transition_relaxation_lindblad_rule,
)
from .._core.backend import _PulseBackend
from .._core.config import _EmulatorConfig
from .._core.lindblad import _lindblad_noise_rejection_reasons
from .._core.outcome import ExecutionMode
from .._core.planning import _PreparedPulseProgram
from .._core.pulse import PulseImplementationMap
from .calibration import default_transmon_calibration
from .model import TransmonModel
from .realization import (
    _validate_transmon_map_compatibility,
    default_transmon_gate_implementation_map,
)
from .target import _LOCAL_DIMENSION, _TransmonTarget


def _default_lindblad_map() -> LindbladImplementationMap:
    """Return a fresh Transmon continuous-noise catalog."""
    implementations = LindbladImplementationMap()
    implementations.add(
        TransitionRelaxation,
        transition_relaxation_lindblad_rule,
    )
    implementations.add(PhaseDamping, phase_damping_lindblad_rule)
    implementations.add(Depolarizing, depolarizing_lindblad_rule)
    return implementations


class TransmonEmulator(_PulseBackend):
    """Run gates and direct controls on a three-level transmon model.

    Every run starts with all model transmons in physical ``|0>``. Ordinary
    gates use a pulse implementation map; ``PulseOperation`` values use their
    channels directly.

    Args:
        model: Transmon model created with ``TransmonModel.from_document``.
        method: Mathematical result representation: ``"statevector"``
            (default), ``"density_matrix"``, or ``"unitary"``. The aliases
            ``"SV"`` and ``"DM"`` are accepted.
        noise: Noise applied by this emulator. The default is no noise.
        gate_implementation_map: Gate-to-pulse rules. ``None`` uses the
            built-in transmon gate map and packaged calibration.

    Raises:
        BackendValidationError: If an argument has the wrong type or ``noise``
            contains a declaration unsupported by the selected rules, or if a
            standard gate map was compiled for an incompatible transmon model.

    Examples:
        >>> import fatqat as fq
        >>> model = fq.emulator.TransmonModel.from_document(
        ...     fq.emulator.load_model_document("transmon.reference")
        ... )
        >>> backend = fq.emulator.TransmonEmulator(model)
        >>> backend.model is model
        True
    """

    def __init__(
        self,
        model: TransmonModel,
        *,
        method: str = "statevector",
        noise: NoiseModel | None = None,
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
        _validate_transmon_map_compatibility(effective_gate_map, model)
        super().__init__(
            model,
            method=method,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=_default_lindblad_map(),
        )
        self.validate_noise_model(self._noise_model)
        self._set_target(_TransmonTarget(model))

    def _noise_model_rejection_reasons(
        self, noise_model: NoiseModel
    ) -> tuple[str, ...]:
        return _lindblad_noise_rejection_reasons(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=_LOCAL_DIMENSION,
            backend_name=type(self).__name__,
            supports_readout_confusion=True,
            readout_confusion_shape=(2, 2),
        )

    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        simulation: _EmulatorConfig,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        del simulation
        from .qutip_adapter import _TransmonQutipAdapter

        return _TransmonQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            execution_mode=execution_mode,
            retain_final_state=retain_final_state,
        )


__all__ = ["TransmonEmulator"]
