"""Two-level neutral-atom pulse emulator."""

from __future__ import annotations

from typing import Any, cast

from ..atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ThermalRelaxation,
)
from ...noise.lindblad import (
    LindbladImplementationMap,
    amplitude_damping_lindblad_rule,
    depolarizing_lindblad_rule,
    phase_damping_lindblad_rule,
    thermal_relaxation_lindblad_rule,
)
from ...operations import Barrier, Measurement, PulseOperation, Reset
from ...program import Program, _AppliedOperation
from .._core.backend import _PulseBackend
from .._core.config import _EmulatorConfig
from .._core.lindblad import _lindblad_noise_rejection_reasons
from .._core.outcome import ExecutionMode
from .._core.planning import _PreparedPulseProgram
from .._core.pulse import PulseImplementationMap
from .config import _Atom2LevelSimulationConfig
from .model import Atom2LevelModel
from .target import _Atom2LevelTarget, _LOCAL_DIMENSION


def _default_lindblad_map() -> LindbladImplementationMap:
    """Return a fresh Atom2 continuous-noise catalog."""
    implementations = LindbladImplementationMap()
    implementations.add(
        AmplitudeDamping,
        amplitude_damping_lindblad_rule,
    )
    implementations.add(
        PhaseDamping,
        phase_damping_lindblad_rule,
    )
    implementations.add(
        ThermalRelaxation,
        thermal_relaxation_lindblad_rule,
    )
    implementations.add(
        Depolarizing,
        depolarizing_lindblad_rule,
    )
    return implementations


class Atom2LevelEmulator(_PulseBackend):
    """Emulate global two-level Rydberg controls on a fixed arrangement.

    Every run starts with each site in ``|g>``. The built-in gate map is empty,
    so programs normally use channel-addressed ``PulseOperation`` values.

    Args:
        model: Two-level atom model created with
            ``Atom2LevelModel.from_document``.
        arrangement: Fixed site coordinates for the program. The program must
            contain one dimension-two resource per site.
        method: Mathematical result representation: ``"statevector"``
            (default), ``"density_matrix"``, or ``"unitary"``. The aliases
            ``"SV"`` and ``"DM"`` are accepted.
        noise: Noise applied by this emulator. The default is no noise.
        gate_implementation_map: Gate-to-pulse rules. ``None`` uses an empty
            map.
    Raises:
        BackendValidationError: If an argument is invalid or ``noise``
            contains a declaration unsupported by the selected rules.

    Examples:
        >>> import fatqat as fq
        >>> model = fq.emulator.Atom2LevelModel.from_document(
        ...     fq.emulator.load_model_document("atom2level.reference")
        ... )
        >>> arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
        >>> backend = fq.emulator.Atom2LevelEmulator(
        ...     model, arrangement=arrangement
        ... )
        >>> backend.arrangement is arrangement
        True
    """

    _simulation_config_cls = _Atom2LevelSimulationConfig

    def __init__(
        self,
        model: Atom2LevelModel,
        *,
        arrangement: AtomArrangement,
        method: str = "statevector",
        noise: NoiseModel | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
    ) -> None:
        if not isinstance(model, Atom2LevelModel):
            raise BackendValidationError("model must be an Atom2LevelModel")
        if not isinstance(arrangement, AtomArrangement):
            raise BackendValidationError("arrangement must be an AtomArrangement")

        self._arrangement = arrangement
        effective_gate_map = (
            PulseImplementationMap()
            if gate_implementation_map is None
            else gate_implementation_map
        )
        super().__init__(
            model,
            method=method,
            noise=noise,
            gate_implementation_map=effective_gate_map,
            lindblad_implementation_map=_default_lindblad_map(),
        )
        self.validate_noise_model(self._noise_model)
        self._set_target(_Atom2LevelTarget(model, arrangement))

    @property
    def arrangement(self) -> AtomArrangement:
        """Return the exact physical site geometry bound to this emulator.

        Returns:
            The arrangement supplied at construction.
        """

        return self._arrangement

    def _validate_source_program(self, program: Program) -> None:
        measurement_suffix = False
        for instruction in program._instructions:
            if isinstance(instruction, Measurement):
                measurement_suffix = True
                continue
            if not isinstance(instruction, _AppliedOperation):
                raise BackendValidationError(
                    "two-level atom program contains an unknown source instruction"
                )
            if instruction.condition is not None:
                raise BackendValidationError(
                    "Atom2LevelEmulator does not support conditioned operations"
                )
            if isinstance(instruction.operation, type(Barrier)):
                continue
            if measurement_suffix:
                raise BackendValidationError(
                    "operations must precede the terminal measurement suffix"
                )
            if isinstance(instruction.operation, type(Reset)):
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

    def _noise_model_rejection_reasons(
        self, noise_model: NoiseModel
    ) -> tuple[str, ...]:
        return _lindblad_noise_rejection_reasons(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=_LOCAL_DIMENSION,
            backend_name=type(self).__name__,
            allow_operation_scoped=False,
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
        from .qutip_adapter import _Atom2LevelQutipAdapter

        atom2_simulation = cast(_Atom2LevelSimulationConfig, simulation)
        return _Atom2LevelQutipAdapter(
            self._target,
            engine_allocation=prepared.engine_allocation,
            interaction_cutoff=atom2_simulation.interaction_cutoff,
            background_noise=prepared.background_noise,
            execution_mode=execution_mode,
            retain_final_state=retain_final_state,
        )


__all__ = ["Atom2LevelEmulator"]
