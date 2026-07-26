"""Private qutip-qip binding for full-qutrit superconducting evolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from qutip import Qobj, basis, ket2dm, mesolve, qeye, tensor
from qutip_qip.pulse import Drift, Pulse

from ...backends.steps import MeasurementStep, ResetStep
from ...errors import BackendValidationError
from ...noise import ContinuousNoise, ThermalRelaxation
from .engine import _ShotContext, _condition_matches
from .execution import _PlacedPulseRun
from .resolved import PhaseShift, PhaseSwap, SampledControl
from .superconducting import ControlChannelRef, CouplingRef, PhysicsModel

_EPSILON = 1e-12
_SOLVER_OPTIONS = {
    "method": "adams",
    "atol": 1e-11,
    "rtol": 1e-9,
    "nsteps": 10000,
}
FRAME_CONVENTION = "per-subsystem near-resonant rotating frames (Delta_i = 0)"


@dataclass(frozen=True)
class _PulseShotResult:
    """Private NumPy/classical payload returned for one completed shot."""

    density_matrix: np.ndarray
    classical_digits: tuple[int, ...]


class SCQutipAdapter:
    """Engine-private model runner using qutip-qip pulse/drift assembly."""

    def __init__(
        self,
        model: PhysicsModel,
        *,
        engine_to_model: tuple[int, ...] | None = None,
        continuous_noise: tuple[tuple[ContinuousNoise, ...], ...] | None = None,
    ) -> None:
        self._model = model
        self._dims = [model.physical_dimension] * len(model.subsystems)
        self._engine_to_model = (
            tuple(range(len(model.subsystems)))
            if engine_to_model is None
            else tuple(engine_to_model)
        )
        if len(set(self._engine_to_model)) != len(self._engine_to_model) or any(
            type(ordinal) is not int or not 0 <= ordinal < len(model.subsystems)
            for ordinal in self._engine_to_model
        ):
            raise BackendValidationError(
                "engine-to-model subsystem mapping must contain unique model ordinals"
            )
        self._local_annihilation = Qobj(model.annihilation)
        self._local_number = Qobj(model.number)
        self._annihilation = tuple(
            self._expand_local(ordinal, self._local_annihilation)
            for ordinal in range(len(model.subsystems))
        )
        self._number = tuple(
            self._expand_local(ordinal, self._local_number)
            for ordinal in range(len(model.subsystems))
        )
        self._drift = self._build_drift()
        noise_by_subsystem = continuous_noise or tuple(() for _ in model.subsystems)
        if len(noise_by_subsystem) != len(model.subsystems):
            raise BackendValidationError(
                "continuous noise must align with every physical model subsystem"
            )
        self._collapse_operators = self._build_continuous_noise(noise_by_subsystem)
        self._projectors = tuple(
            tuple(
                self._expand_local(
                    ordinal, ket2dm(basis(model.physical_dimension, level))
                )
                for level in range(model.physical_dimension)
            )
            for ordinal in range(len(model.subsystems))
        )
        self._reset_operators = tuple(
            tuple(
                self._expand_local(
                    ordinal,
                    basis(model.physical_dimension, 0)
                    * basis(model.physical_dimension, level).dag(),
                )
                for level in range(model.physical_dimension)
            )
            for ordinal in range(len(model.subsystems))
        )

    @staticmethod
    def solver_metadata() -> dict[str, Any]:
        """Return normalized, public-safe solver facts for result metadata."""
        return {
            "frame_convention": FRAME_CONVENTION,
            "options": dict(_SOLVER_OPTIONS),
        }

    def initial_state(self) -> Any:
        """Create the full-model physical ground-state density matrix."""
        ket = tensor(*(basis(dimension, 0) for dimension in self._dims))
        return ket2dm(ket)

    def evolve(
        self,
        run: _PlacedPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None:
        """Evolve one placed region and commit its enabled post-frame actions."""
        if len(enabled) != len(run.blocks):
            raise BackendValidationError(
                "pulse enable flags must align with the placed run"
            )
        if run.start_ns < context.time_ns - _EPSILON:
            raise BackendValidationError("placed pulse runs must be time ordered")

        frames = dict(context.frame_angles)
        pulses: list[Pulse] = []
        pending_actions: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]] = (
            []
        )
        ordered = sorted(
            range(len(run.blocks)),
            key=lambda index: (run.starts_ns[index], index),
        )
        for source_index in ordered:
            block = run.blocks[source_index]
            start_ns = run.starts_ns[source_index]
            self._apply_ready_actions(pending_actions, start_ns, frames)
            if not enabled[source_index]:
                continue
            for child in block.children:
                pulses.append(self._bind_child(child, start_ns, frames))
            pending_actions.append(
                (
                    start_ns + block.duration_ns,
                    source_index,
                    block.post_actions,
                )
            )

        state = context.state
        if run.end_ns > context.time_ns + _EPSILON:
            hamiltonian = self._drift.get_ideal_qobjevo(self._dims)
            for pulse in pulses:
                contribution, collapse = pulse.get_noisy_qobjevo(self._dims)
                if collapse:
                    raise BackendValidationError(
                        "ideal pulse binding unexpectedly produced collapse terms"
                    )
                hamiltonian += contribution
            result = mesolve(
                hamiltonian,
                state,
                [context.time_ns, run.end_ns],
                c_ops=self._collapse_operators,
                options=_SOLVER_OPTIONS,
            )
            state = result.states[-1]

        for _end, _source, actions in sorted(
            pending_actions, key=lambda event: event[:2]
        ):
            self._apply_actions(actions, frames)
        context.state = state
        context.frame_angles.clear()
        context.frame_angles.update(frames)

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None:
        """Execute physical qutrit measurement or guarded reset."""
        if isinstance(step, MeasurementStep):
            maps = step.reported_digit_maps or tuple(
                tuple(range(self._model.physical_dimension))
                for _ in step.measured_indices
            )
            confusions = step.confusions or (None,) * len(step.measured_indices)
            for engine_index, classical_index, digit_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                maps,
                confusions,
            ):
                outcome = self._measure(self._model_ordinal(engine_index), context)
                reported = digit_map[outcome]
                if confusion is not None:
                    reported = int(
                        context.rng.choice(len(confusion), p=confusion[:, reported])
                    )
                context.classical_memory[classical_index] = reported
            return
        if _condition_matches(step.condition, context.classical_memory):
            for engine_index in step.reset_indices:
                self._reset(self._model_ordinal(engine_index), context)

    def finish_shot(self, context: _ShotContext) -> _PulseShotResult:
        """Return a NumPy copy; no solver value crosses the engine boundary."""
        return _PulseShotResult(
            density_matrix=np.array(context.state.full(), dtype=complex, copy=True),
            classical_digits=tuple(context.classical_memory),
        )

    def _expand_local(self, ordinal: int, operator: Any) -> Any:
        factors = [qeye(dimension) for dimension in self._dims]
        factors[ordinal] = operator
        return tensor(*factors)

    def _build_drift(self) -> Drift:
        drift = Drift()
        identity = qeye(self._model.physical_dimension)
        for ordinal, subsystem in enumerate(self._model.subsystems):
            local = (
                2
                * pi
                * subsystem.anharmonicity_ghz
                * self._local_number
                * (self._local_number - identity)
                / 2
            )
            drift.add_drift(local, ordinal)
        for coupling in self._model.couplings:
            if coupling.residual_exchange_ghz == 0.0:
                continue
            exchange = tensor(
                self._local_annihilation.dag(), self._local_annihilation
            ) + tensor(self._local_annihilation, self._local_annihilation.dag())
            targets = [
                self._model.subsystem_ids.index(identifier)
                for identifier in coupling.subsystem_ids
            ]
            drift.add_drift(2 * pi * coupling.residual_exchange_ghz * exchange, targets)
        return drift

    def _build_continuous_noise(
        self, noise_by_subsystem: tuple[tuple[ContinuousNoise, ...], ...]
    ) -> tuple[Any, ...]:
        noise_pulse = Pulse(None, None)
        for ordinal, sources in enumerate(noise_by_subsystem):
            for source in sources:
                if not isinstance(source, ThermalRelaxation):
                    raise BackendValidationError(
                        f"{type(source).__name__} is not supported by the pulse backend"
                    )
                noise_pulse.add_lindblad_noise(
                    sqrt(1 / source.T1_ns) * self._local_annihilation,
                    ordinal,
                    coeff=True,
                )
                if source.T2_ns < 2 * source.T1_ns:
                    pure_dephasing = 2 * (1 / source.T2_ns - 1 / (2 * source.T1_ns))
                    noise_pulse.add_lindblad_noise(
                        sqrt(pure_dephasing) * self._local_number,
                        ordinal,
                        coeff=True,
                    )
        if not noise_pulse.lindblad_noise:
            return ()
        _zero, collapse = noise_pulse.get_noisy_qobjevo(self._dims)
        return tuple(collapse)

    def _model_ordinal(self, engine_index: int) -> int:
        try:
            return self._engine_to_model[engine_index]
        except (IndexError, TypeError):
            raise BackendValidationError(
                f"unknown pulse-engine subsystem index {engine_index!r}"
            ) from None

    def _measure(self, ordinal: int, context: _ShotContext) -> int:
        probabilities = np.array(
            [
                max(0.0, float(np.real((projector * context.state).tr())))
                for projector in self._projectors[ordinal]
            ]
        )
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("physical measurement produced invalid probabilities")
        probabilities /= total
        outcome = int(context.rng.choice(len(probabilities), p=probabilities))
        projector = self._projectors[ordinal][outcome]
        probability = probabilities[outcome]
        context.state = projector * context.state * projector / probability
        return outcome

    def _reset(self, ordinal: int, context: _ShotContext) -> None:
        context.state = sum(
            operator * context.state * operator.dag()
            for operator in self._reset_operators[ordinal]
        )

    def _bind_child(
        self,
        child: SampledControl,
        block_start_ns: float,
        frames: dict[Any, float],
    ) -> Pulse:
        absolute_tlist = (
            block_start_ns + child.start_offset_ns + np.asarray(child.tlist)
        )
        coefficients = np.asarray(child.coefficients)
        channel = child.channel
        if isinstance(channel, ControlChannelRef):
            ordinal = self._model.bind_control(channel)
            if channel.kind == "detuning":
                real = self._require_real(coefficients, "detuning")
                return Pulse(
                    self._local_number,
                    ordinal,
                    tlist=absolute_tlist,
                    coeff=real,
                    spline_kind="cubic",
                    label="detuning",
                )
            phase = np.exp(
                1j
                * frames.get(self._model.frame(self._model.subsystem_ids[ordinal]), 0.0)
            )
            envelope = phase * coefficients
            x_operator = self._local_annihilation + self._local_annihilation.dag()
            y_operator = -1j * (
                self._local_annihilation - self._local_annihilation.dag()
            )
            pulse = Pulse(
                x_operator,
                ordinal,
                tlist=absolute_tlist,
                coeff=envelope.real,
                spline_kind="cubic",
                label="drive",
            )
            pulse.add_coherent_noise(
                y_operator,
                ordinal,
                tlist=absolute_tlist,
                coeff=envelope.imag,
            )
            return pulse
        if isinstance(channel, CouplingRef):
            coupling = self._model.couplings[self._model.bind_coupling(channel)]
            targets = [
                self._model.subsystem_ids.index(identifier)
                for identifier in coupling.subsystem_ids
            ]
            exchange = tensor(
                self._local_annihilation.dag(), self._local_annihilation
            ) + tensor(self._local_annihilation, self._local_annihilation.dag())
            return Pulse(
                exchange,
                targets,
                tlist=absolute_tlist,
                coeff=self._require_real(coefficients, "exchange"),
                spline_kind="cubic",
                label="exchange",
            )
        raise BackendValidationError("pulse control has an unknown channel reference")

    @staticmethod
    def _require_real(coefficients: np.ndarray, name: str) -> np.ndarray:
        if not np.allclose(coefficients.imag, 0.0, atol=1e-12, rtol=0.0):
            raise BackendValidationError(f"{name} pulse coefficients must be real")
        return np.asarray(coefficients.real)

    @classmethod
    def _apply_ready_actions(
        cls,
        events: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]],
        start_ns: float,
        frames: dict[Any, float],
    ) -> None:
        ready = [event for event in events if event[0] <= start_ns + _EPSILON]
        for _end, _source, actions in sorted(ready, key=lambda event: event[:2]):
            cls._apply_actions(actions, frames)
        events[:] = [event for event in events if event not in ready]

    @staticmethod
    def _apply_actions(
        actions: tuple[PhaseShift | PhaseSwap, ...], frames: dict[Any, float]
    ) -> None:
        for action in actions:
            if isinstance(action, PhaseShift):
                frames[action.frame] = frames.get(action.frame, 0.0) + action.angle_rad
            else:
                first = frames.get(action.first, 0.0)
                second = frames.get(action.second, 0.0)
                frames[action.first], frames[action.second] = second, first
