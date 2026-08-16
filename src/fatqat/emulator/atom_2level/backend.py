"""Two-level neutral-atom pulse emulator."""

from __future__ import annotations

from typing import Any

from ...atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...noise import (
    AmplitudeDamping,
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    PhaseDamping,
)
from ...noise.lindblad import (
    amplitude_damping_lindblad_rule,
    phase_damping_lindblad_rule,
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
from .policy import GridInteractionPolicy
from .target import _Atom2LevelTarget


def _default_lindblad_map() -> LindbladImplementationMap:
    implementations = LindbladImplementationMap()
    implementations.register(
        AmplitudeDamping,
        amplitude_damping_lindblad_rule,
    )
    implementations.register(
        PhaseDamping,
        phase_damping_lindblad_rule,
    )
    return implementations


class Atom2LevelEmulator(_PulseBackend):
    """Two-level neutral-atom emulator with global Rydberg controls."""

    _coherent_execution_mode: ExecutionMode = "statevector"

    def __init__(
        self,
        model: Atom2LevelModel,
        *,
        arrangement: AtomArrangement,
        interaction_policy: GridInteractionPolicy | None = None,
        noise: NoiseModel | None = None,
        gate_implementation_map: PulseImplementationMap | None = None,
        lindblad_implementation_map: LindbladImplementationMap | None = None,
    ) -> None:
        if not isinstance(model, Atom2LevelModel):
            raise BackendValidationError("model must be an Atom2LevelModel")
        if not isinstance(arrangement, AtomArrangement):
            raise BackendValidationError("arrangement must be an AtomArrangement")
        policy = (
            GridInteractionPolicy.nearest_neighbor()
            if interaction_policy is None
            else interaction_policy
        )
        if not isinstance(policy, GridInteractionPolicy):
            raise BackendValidationError(
                "interaction_policy must be a GridInteractionPolicy or None"
            )

        self._arrangement = arrangement
        self._interaction_policy = policy
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
        report = self._classify_noise(self._noise_model)
        if not report.supported:
            raise BackendValidationError("; ".join(report.warnings))
        self._set_target(_Atom2LevelTarget(model, arrangement, policy))

    @property
    def arrangement(self) -> AtomArrangement:
        """Return the arrangement bound to this emulator."""

        return self._arrangement

    @property
    def interaction_policy(self) -> GridInteractionPolicy:
        """Return the interaction policy bound to this emulator."""

        return self._interaction_policy

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
            supports_readout_confusion=False,
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
