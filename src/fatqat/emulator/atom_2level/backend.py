"""Two-level neutral-atom pulse emulator."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

from ..atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...noise import (
    AmplitudeDamping,
    Depolarizing,
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    PhaseDamping,
    ThermalRelaxation,
)
from ...noise.lindblad import (
    amplitude_damping_lindblad_rule,
    depolarizing_lindblad_rule,
    phase_damping_lindblad_rule,
    thermal_relaxation_lindblad_rule,
)
from ...operations import BarrierGate, Measurement, PulseOperation, ResetGate
from ...program import AppliedOperation, Program
from .._core.backend import _PulseBackend
from .._core.lindblad import _classify_lindblad_noise
from .._core.outcome import ExecutionMode
from .._core.planning import (
    PulsePlanFacts,
    _PreparedPulseProgram,
)
from .._core.pulse import PulseImplementationMap
from .model import Atom2LevelModel
from .target import _Atom2LevelTarget

_CUTOFF_ERROR = "interaction_cutoff must be None or a finite nonnegative real number"


def _normalize_interaction_cutoff(value: object) -> float | None:
    """Normalize the public distance cutoff at the backend boundary."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BackendValidationError(_CUTOFF_ERROR)
    try:
        cutoff = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendValidationError(_CUTOFF_ERROR) from exc
    if not isfinite(cutoff) or cutoff < 0.0:
        raise BackendValidationError(_CUTOFF_ERROR)
    return cutoff


def _default_lindblad_map() -> LindbladImplementationMap:
    """Return a fresh Atom2-only built-in continuous-noise catalog.

    This map is used only when the caller omits a map; an explicitly supplied
    map replaces the catalog rather than extending it implicitly.
    """
    implementations = LindbladImplementationMap()
    implementations.register(
        AmplitudeDamping,
        amplitude_damping_lindblad_rule,
    )
    implementations.register(
        PhaseDamping,
        phase_damping_lindblad_rule,
    )
    implementations.register(
        ThermalRelaxation,
        thermal_relaxation_lindblad_rule,
    )
    implementations.register(
        Depolarizing,
        depolarizing_lindblad_rule,
    )
    return implementations


class Atom2LevelEmulator(_PulseBackend):
    """Emulate global two-level Rydberg controls on a fixed arrangement.

    Args:
        model: Validated two-level atom physics model.
        arrangement: Exact physical site geometry for the pulse program.
        interaction_cutoff: Maximum interacting-pair separation in the
            model's distance unit. ``None`` retains every unordered pair and
            ``0.0`` disables pair interactions.
        noise: Optional authored noise model.
        gate_implementation_map: Optional gate-to-pulse implementations.
        lindblad_implementation_map: Optional continuous-noise rules.

    Raises:
        BackendValidationError: If an input is invalid or unsupported.
    """

    _coherent_execution_mode: ExecutionMode = "statevector"

    def __init__(
        self,
        model: Atom2LevelModel,
        *,
        arrangement: AtomArrangement,
        interaction_cutoff: float | None = None,
        noise: NoiseModel | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
    ) -> None:
        if not isinstance(model, Atom2LevelModel):
            raise BackendValidationError("model must be an Atom2LevelModel")
        if not isinstance(arrangement, AtomArrangement):
            raise BackendValidationError("arrangement must be an AtomArrangement")
        cutoff = _normalize_interaction_cutoff(interaction_cutoff)

        self._arrangement = arrangement
        self._interaction_cutoff = cutoff
        self._uses_builtin_lindblad_defaults = lindblad_implementation_map is None
        effective_gate_map = (
            PulseImplementationMap()
            if gate_implementation_map is None
            else gate_implementation_map
        )
        effective_lindblad_map = (
            _default_lindblad_map()
            if lindblad_implementation_map is None
            else lindblad_implementation_map
        )
        super().__init__(
            model,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=effective_lindblad_map,
        )
        self._require_captured_noise_support()
        self._set_target(_Atom2LevelTarget(model, arrangement, cutoff))

    @property
    def arrangement(self) -> AtomArrangement:
        """Return the exact physical site geometry bound to this emulator.

        Returns:
            The immutable arrangement supplied at construction.
        """

        return self._arrangement

    @property
    def interaction_cutoff(self) -> float | None:
        """Return the normalized maximum interaction-pair distance.

        Returns:
            ``None`` when all unordered pairs are retained, otherwise the
            finite nonnegative cutoff in the model's distance unit.
        """

        return self._interaction_cutoff

    def _validate_source_program(self, program: Program) -> None:
        measurement_suffix = False
        for instruction in program.operations:
            if isinstance(instruction, Measurement):
                measurement_suffix = True
                continue
            if not isinstance(instruction, AppliedOperation):
                raise BackendValidationError(
                    "two-level atom program contains an unknown source instruction"
                )
            if instruction.condition is not None:
                raise BackendValidationError(
                    "Atom2LevelEmulator does not support conditioned operations"
                )
            if isinstance(instruction.operation, BarrierGate):
                continue
            if measurement_suffix:
                raise BackendValidationError(
                    "operations must precede the terminal measurement suffix"
                )
            if isinstance(instruction.operation, ResetGate):
                raise BackendValidationError(
                    "Atom2LevelEmulator does not support reset"
                )
            if (
                isinstance(instruction.operation, PulseOperation)
                and instruction.targets
            ):
                raise BackendValidationError(
                    "Atom2LevelEmulator global pulses do not accept targets"
                )

    def _classify_noise(self, noise_model: NoiseModel) -> NoiseSupportReport:
        return _classify_lindblad_noise(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=self.model.local_dimension,
            backend_name=type(self).__name__,
            allow_operation_scoped=not self._uses_builtin_lindblad_defaults,
            supports_readout_confusion=True,
            readout_confusion_shape=(2, 2),
        )

    def _resolve_execution_mode(self, facts: PulsePlanFacts) -> ExecutionMode:
        has_lindblad = (
            facts.has_resolved_lindblad
            or facts.has_supported_background_lindblad_registration
        )
        if has_lindblad and facts.has_measurement:
            return "trajectory" if facts.has_nonzero_evolution else "statevector"
        if has_lindblad:
            return "density_matrix"
        return "statevector"

    def _create_runner(
        self,
        prepared: _PreparedPulseProgram,
        *,
        execution_mode: ExecutionMode,
        retain_final_state: bool,
    ) -> Any:
        from .qutip_adapter import _Atom2LevelQutipAdapter

        return _Atom2LevelQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            execution_mode=execution_mode,
            retain_final_state=retain_final_state,
        )


__all__ = ["Atom2LevelEmulator"]
