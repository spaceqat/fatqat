"""Private qutip-qip binding for full-qutrit superconducting evolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from qutip import (
    Qobj,
    basis,
    ket2dm,
    mesolve,
    propagator as qutip_propagator,
    qeye,
    tensor,
)
from qutip_qip.pulse import Drift, Pulse

from ..backends.steps import MeasurementStep, ResetStep
from ..errors import BackendValidationError
from .engine import _ShotContext, _condition_matches
from .scheduling import _ScheduledPulseRun
from .pulse import PhaseShift, PhaseSwap, PulseBlock, SampledControl
from .lindblad import ResolvedLindbladTerm
from .superconducting import ControlChannelRef, PhysicsModel

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


@dataclass(frozen=True)
class _BoundFrames:
    """Binding result for a zero-duration, virtual-frame-only region."""

    output_frames: dict[Any, float]


@dataclass(frozen=True)
class _BoundDynamics:
    """Binding result for nonzero QuTiP dynamics and its output frames."""

    hamiltonian: Any
    collapse_operators: tuple[Any, ...]
    output_frames: dict[Any, float]


class SCQutipAdapter:
    """Internal qutip-qip runner for full-qutrit transmon evolution.

    The adapter is the only emulator layer that constructs QuTiP values. It
    tensor-expands model-local operators, binds scheduled sampled controls and
    Lindblad terms, integrates open-system density matrices, constructs
    coherent propagators, and implements physical measurement/reset. The
    :class:`PulseEngine` owns shots, conditions, traversal, and scheduling.

    Inputs and completed-shot outputs use fatqat/NumPy values, so no QuTiP
    object crosses the backend's public boundary. This class is intentionally
    not exported from :mod:`fatqat.backends`.
    """

    def __init__(
        self,
        model: PhysicsModel,
        *,
        engine_index_to_model_ordinal: tuple[int, ...] | None = None,
        always_on_noise: tuple[ResolvedLindbladTerm, ...] = (),
    ) -> None:
        """Bind one immutable physical model to the QuTiP execution layer.

        ``engine_index_to_model_ordinal`` maps only program-addressable engine
        indices. The full model still participates in drift, noise, state, and
        output dimensions. ``always_on_noise`` contains already resolved local
        collapse terms; the adapter never interprets source noise descriptors.
        """
        self._model = model
        self._dims = [model.physical_dimension] * len(model.subsystems)
        self._engine_index_to_model_ordinal = (
            tuple(range(len(model.subsystems)))
            if engine_index_to_model_ordinal is None
            else tuple(engine_index_to_model_ordinal)
        )
        if len(set(self._engine_index_to_model_ordinal)) != len(
            self._engine_index_to_model_ordinal
        ) or any(
            type(ordinal) is not int or not 0 <= ordinal < len(model.subsystems)
            for ordinal in self._engine_index_to_model_ordinal
        ):
            raise BackendValidationError(
                "engine-index-to-model-ordinal mapping must contain unique model ordinals"
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
        self._collapse_operators = self._build_always_on_noise(always_on_noise)
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

    @staticmethod
    def copy_state(state: Any) -> Any:
        """Copy a state for an independent terminal-measurement trajectory."""
        return state.copy()

    def evolve(
        self,
        run: _ScheduledPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None:
        """Evolve one placed region and commit enabled post-frame actions.

        False enable flags suppress a block's controls, operation-scoped noise,
        and post-actions while elapsed region time and always-on dynamics
        remain active.
        """
        bound = self._bind_run(
            run,
            enabled=enabled,
            input_time=context.time,
            input_frames=context.frame_angles,
        )
        state = context.state
        if isinstance(bound, _BoundDynamics):
            result = mesolve(
                bound.hamiltonian,
                state,
                [context.time, run.end_time],
                c_ops=bound.collapse_operators,
                options=_SOLVER_OPTIONS,
            )
            state = result.states[-1]

        context.state = state
        context.frame_angles.clear()
        context.frame_angles.update(bound.output_frames)

    def propagator(
        self, run: _ScheduledPulseRun, *, apply_final_frame: bool = True
    ) -> Any:
        """Return the coherent full-model propagator for one scheduled run.

        Intermediate frame updates always affect later pulse binding.
        ``apply_final_frame`` controls only whether the output frame ledger is
        composed onto the returned Hamiltonian-generated propagator.
        """
        if abs(run.start_time) > _EPSILON:
            raise BackendValidationError("a propagator run must start at time zero")
        bound = self._bind_run(
            run,
            enabled=(True,) * len(run.blocks),
            input_time=0.0,
            input_frames={},
        )
        if isinstance(bound, _BoundFrames):
            unitary = tensor(*(qeye(dimension) for dimension in self._dims))
        elif bound.collapse_operators:
            raise BackendValidationError(
                "propagator is unavailable for dissipative pulse evolution"
            )
        else:
            unitary = qutip_propagator(
                bound.hamiltonian,
                run.end_time,
                options=_SOLVER_OPTIONS,
            )
        if not apply_final_frame:
            return unitary
        return self._frame_unitary(bound.output_frames) * unitary

    def _bind_run(
        self,
        run: _ScheduledPulseRun,
        *,
        enabled: tuple[bool, ...],
        input_time: float,
        input_frames: dict[Any, float],
    ) -> _BoundFrames | _BoundDynamics:
        """Bind a run as either frame-only or nonzero QuTiP dynamics."""
        if len(enabled) != len(run.blocks):
            raise BackendValidationError(
                "pulse enable flags must align with the placed run"
            )
        if run.start_time < input_time - _EPSILON:
            raise BackendValidationError("placed pulse runs must be time ordered")

        frames = dict(input_frames)
        pulses: list[Pulse] = []
        noise_pulses: list[Pulse] = []
        pending_actions: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]] = (
            []
        )
        ordered = sorted(
            range(len(run.blocks)),
            key=lambda index: (run.starts[index], index),
        )
        for source_index in ordered:
            block = run.blocks[source_index]
            start_time = run.starts[source_index]
            self._apply_ready_actions(pending_actions, start_time, frames)
            if not enabled[source_index]:
                continue
            for child in block.controls:
                pulses.append(self._bind_child(child, start_time, frames))
            # A zero-duration block cannot contribute noise: even a
            # rate-mode descriptor's effect over zero time is a no-op, and a
            # nonzero-probability one was already rejected at lowering.
            if block.noise and block.duration > 0.0:
                noise_pulses.append(
                    self._bind_block_noise(block, start_time, input_time, run.end_time)
                )
            pending_actions.append(
                (
                    start_time + block.duration,
                    source_index,
                    block.post_actions,
                )
            )

        for _end, _source, actions in sorted(
            pending_actions, key=lambda event: event[:2]
        ):
            self._apply_actions(actions, frames)
        if run.end_time <= input_time + _EPSILON:
            return _BoundFrames(output_frames=frames)

        hamiltonian = self._drift.get_ideal_qobjevo(self._dims)
        for pulse in pulses:
            contribution, collapse = pulse.get_noisy_qobjevo(self._dims)
            if collapse:
                raise BackendValidationError(
                    "ideal pulse binding unexpectedly produced collapse terms"
                )
            hamiltonian += contribution
        local_collapse: list[Any] = []
        for noise_pulse in noise_pulses:
            _zero, collapse = noise_pulse.get_noisy_qobjevo(self._dims)
            local_collapse.extend(collapse)

        return _BoundDynamics(
            hamiltonian=hamiltonian,
            collapse_operators=tuple(self._collapse_operators) + tuple(local_collapse),
            output_frames=frames,
        )

    def _frame_unitary(self, frames: dict[Any, float]) -> Any:
        """Build the full-model basis transform for terminal frame angles."""
        factors = []
        levels = np.arange(self._model.physical_dimension)
        for subsystem_id in self._model.subsystem_ids:
            angle = frames.get(self._model.frame(subsystem_id), 0.0)
            factors.append(Qobj(np.diag(np.exp(1j * angle * levels))))
        return tensor(*factors)

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None:
        """Execute physical qutrit measurement or guarded reset.

        Measurement samples and collapses each addressed qutrit, maps its
        physical level to the lowered reported digit, optionally samples
        readout confusion, and writes classical memory. Reset applies the
        deterministic qutrit-to-ground channel when its condition matches.
        """
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
        """Tensor-expand one local operator at a model subsystem ordinal."""
        factors = [qeye(dimension) for dimension in self._dims]
        factors[ordinal] = operator
        return tensor(*factors)

    def _build_drift(self) -> Drift:
        """Build the rotating-frame local-anharmonicity drift.

        Declared coupling edges contribute no static exchange; exchange enters
        only through an explicitly driven control pulse.
        """
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
        return drift

    def _build_always_on_noise(
        self, bindings: tuple[ResolvedLindbladTerm, ...]
    ) -> tuple[Any, ...]:
        """Build constant collapse terms from resolved always-on bindings."""
        noise_pulse = Pulse(None, None)
        for binding in bindings:
            for local_qobj, ordinal in self._lindblad_ops(binding):
                noise_pulse.add_lindblad_noise(
                    local_qobj,
                    ordinal,
                    coeff=True,
                )
        if not noise_pulse.lindblad_noise:
            return ()
        _zero, collapse = noise_pulse.get_noisy_qobjevo(self._dims)
        return tuple(collapse)

    def _bind_block_noise(
        self,
        block: PulseBlock,
        start_time: float,
        run_start_time: float,
        run_end_time: float,
    ) -> Pulse:
        """Build one block-owned `Pulse` carrying its gate-keyed collapse terms.

        Interval scoping reuses the control path's own time-windowed-pulse
        mechanism rather than splitting the run into multiple `mesolve`
        calls: a step-function coefficient spans the whole solved run
        (``[run_start_time, run_end_time]``) but is 1 only during this block's
        own placed ``[start_time, start_time + duration)`` window, so the
        collapse operator this pulse contributes is on only there.
        """
        end_time = start_time + block.duration
        tlist = np.asarray(
            sorted({run_start_time, start_time, end_time, run_end_time}), dtype=float
        )
        window = np.array(
            [1.0 if start_time <= point < end_time else 0.0 for point in tlist],
            dtype=float,
        )
        noise_pulse = Pulse(None, None)
        for binding in block.noise:
            for local_qobj, ordinal in self._lindblad_ops(binding):
                noise_pulse.add_lindblad_noise(
                    local_qobj, ordinal, tlist=tlist, coeff=window
                )
        return noise_pulse

    def _lindblad_ops(self, term: ResolvedLindbladTerm) -> list[tuple[Any, int]]:
        """Adapt one backend-neutral Lindblad term to local QuTiP operators."""
        local_qobj = Qobj(term.local_operator)
        return [
            (
                local_qobj,
                self._validate_noise_ordinal(ordinal),
            )
            for ordinal in term.model_ordinals
        ]

    def _validate_noise_ordinal(self, ordinal: int) -> int:
        """Validate and return one physical-model subsystem ordinal."""
        if type(ordinal) is not int or not 0 <= ordinal < len(self._model.subsystems):
            raise BackendValidationError(
                f"unknown physical-model subsystem ordinal {ordinal!r}"
            )
        return ordinal

    def _model_ordinal(self, engine_index: int) -> int:
        """Translate a program engine index to its bound model ordinal."""
        try:
            return self._engine_index_to_model_ordinal[engine_index]
        except (IndexError, TypeError):
            raise BackendValidationError(
                f"unknown pulse-engine subsystem index {engine_index!r}"
            ) from None

    def _measure(self, ordinal: int, context: _ShotContext) -> int:
        """Sample one physical qutrit projector and collapse the density matrix."""
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
        """Apply the deterministic local qutrit reset channel."""
        context.state = sum(
            operator * context.state * operator.dag()
            for operator in self._reset_operators[ordinal]
        )

    def _bind_child(
        self,
        child: SampledControl,
        block_start_time: float,
        frames: dict[Any, float],
    ) -> Pulse:
        """Convert one sampled control to an absolute-time qutip-qip pulse.

        Drive controls bind complex envelopes to X/Y quadratures and rotate by
        the negative accumulated virtual-frame angle. Detuning binds the local
        number operator; exchange binds the two-qutrit ladder interaction on
        its declared coupling edge.
        """
        absolute_tlist = block_start_time + child.start_offset + np.asarray(child.tlist)
        coefficients = np.asarray(child.coefficients)
        channel = child.channel
        if not isinstance(channel, ControlChannelRef):
            raise BackendValidationError(
                "pulse control has an unknown channel reference"
            )
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
        if channel.kind == "exchange":
            coupling = self._model.couplings[ordinal]
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
        # `bind_control` above already rejected any kind other than
        # drive/detuning/exchange, so only "drive" remains here.
        phase = np.exp(
            -1j * frames.get(self._model.frame(self._model.subsystem_ids[ordinal]), 0.0)
        )
        envelope = phase * coefficients
        x_operator = self._local_annihilation + self._local_annihilation.dag()
        y_operator = -1j * (self._local_annihilation - self._local_annihilation.dag())
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

    @staticmethod
    def _require_real(coefficients: np.ndarray, name: str) -> np.ndarray:
        """Return real coefficients or reject an imaginary component."""
        if not np.allclose(coefficients.imag, 0.0, atol=1e-12, rtol=0.0):
            raise BackendValidationError(f"{name} pulse coefficients must be real")
        return np.asarray(coefficients.real)

    @classmethod
    def _apply_ready_actions(
        cls,
        events: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]],
        start_time: float,
        frames: dict[Any, float],
    ) -> None:
        """Apply completed frame events before binding a later-time control."""
        ready = [event for event in events if event[0] <= start_time + _EPSILON]
        for _end, _source, actions in sorted(ready, key=lambda event: event[:2]):
            cls._apply_actions(actions, frames)
        events[:] = [event for event in events if event not in ready]

    @staticmethod
    def _apply_actions(
        actions: tuple[PhaseShift | PhaseSwap, ...], frames: dict[Any, float]
    ) -> None:
        """Apply phase shifts/swaps to a mutable virtual-frame ledger."""
        for action in actions:
            if isinstance(action, PhaseShift):
                frames[action.frame] = frames.get(action.frame, 0.0) + action.angle_rad
            else:
                first = frames.get(action.first, 0.0)
                second = frames.get(action.second, 0.0)
                frames[action.first], frames[action.second] = second, first
