"""Private qutip-qip binding for full-qutrit superconducting evolution."""

from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
from qutip import Qobj, basis, ket2dm, mesolve, qeye, tensor
from qutip_qip.pulse import Drift, Pulse

from ...backends.steps import MeasurementStep, ResetStep
from ...errors import BackendValidationError
from .engine import _ShotContext
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


class SCQutipAdapter:
    """Engine-private model runner using qutip-qip pulse/drift assembly."""

    def __init__(self, model: PhysicsModel) -> None:
        self._model = model
        self._dims = [model.physical_dimension] * len(model.subsystems)
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
                c_ops=(),
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
        """Task 7 supplies physical qutrit measurement and reset."""
        del step, context
        raise RuntimeError(
            "physical pulse measurement and reset are not implemented yet"
        )

    def finish_shot(self, context: _ShotContext) -> np.ndarray:
        """Return a NumPy copy; no solver value crosses the engine boundary."""
        return np.array(context.state.full(), dtype=complex, copy=True)

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
