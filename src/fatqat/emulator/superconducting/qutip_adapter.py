"""Private qutip-qip binding for full-qutrit superconducting evolution."""

from __future__ import annotations

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

from ..._backends.steps import MeasurementStep, ResetStep
from ..._index_allocation import _EngineAllocation
from ..._pulse_values import PulseControl
from ...errors import BackendValidationError
from .._core.adapter_common import (
    _BoundDynamics,
    _BoundFrames,
    apply_actions,
    apply_ready_actions,
)
from .._core.engine import _ShotContext, _condition_matches
from .._core.lindblad import ResolvedLindbladTerm
from .._core.outcome import _PulseShotOutcome
from .._core.pulse import PhaseShift, PhaseSwap, PulseBlock
from .._core.scheduling import _ScheduledPulseRun
from .._core.target import _PreparedControlBinding
from .._core.value_validation import TIME_EPSILON
from .model import angular_rate_from_ghz
from .target import _TransmonTarget

_SOLVER_OPTIONS = {
    "method": "adams",
    "atol": 1e-11,
    "rtol": 1e-9,
    "nsteps": 10000,
}
FRAME_CONVENTION = "per-subsystem near-resonant rotating frames (Delta_i = 0)"


class _TransmonQutipAdapter:
    """Internal qutip-qip runner for full-qutrit transmon evolution.

    The adapter is the only emulator layer that constructs QuTiP values. It
    tensor-expands model-local operators, binds scheduled sampled controls and
    Lindblad terms, integrates open-system density matrices, constructs
    coherent propagators, and implements physical measurement/reset. The
    :class:`PulseEngine` owns shots, conditions, traversal, and scheduling.

    Inputs and completed-shot outputs use fatqat/NumPy values, so no QuTiP
    object crosses the backend's public boundary. This class is intentionally
    not exported from :mod:`fatqat.emulator`.
    """

    def __init__(
        self,
        target: _TransmonTarget,
        *,
        engine_allocation: _EngineAllocation,
        always_on_noise: tuple[ResolvedLindbladTerm, ...] = (),
        retain_final_state: bool = True,
    ) -> None:
        """Adapt one already-bound physical target to the QuTiP layer.

        The engine allocation fixes full-model tensor order. ``always_on_noise``
        contains already resolved local collapse terms; the adapter never
        interprets source noise descriptors.
        """
        if not isinstance(target, _TransmonTarget):
            raise BackendValidationError("transmon adapter requires a transmon target")
        self._target = target
        if type(retain_final_state) is not bool:
            raise BackendValidationError("retain_final_state must be a bool")
        self._retain_final_state = retain_final_state
        self._solver_used = "none"
        self._always_on_noise = tuple(always_on_noise)
        self._collapse_operators: tuple[Any, ...] | None = None
        expected_operands = tuple(self._target.model.subsystem_ids)
        if (
            engine_allocation.device_operands != expected_operands
            or engine_allocation.system_dims
            != (target.local_dimension,) * len(expected_operands)
        ):
            raise BackendValidationError(
                "transmon engine allocation must cover the complete model in model "
                "order"
            )
        self._engine_allocation = engine_allocation
        self._dims = list(engine_allocation.system_dims)
        self._local_annihilation = Qobj(self._target.model.annihilation)
        self._local_number = Qobj(self._target.model.number)
        self._annihilation = tuple(
            self._expand_local(ordinal, self._local_annihilation)
            for ordinal in range(len(self._target.model.subsystems))
        )
        self._number = tuple(
            self._expand_local(ordinal, self._local_number)
            for ordinal in range(len(self._target.model.subsystems))
        )
        self._drift = self._build_drift()
        self._projectors = tuple(
            tuple(
                self._expand_local(
                    ordinal, ket2dm(basis(target.local_dimension, level))
                )
                for level in range(target.local_dimension)
            )
            for ordinal in range(len(self._target.model.subsystems))
        )
        self._reset_operators = tuple(
            tuple(
                self._expand_local(
                    ordinal,
                    basis(target.local_dimension, 0)
                    * basis(target.local_dimension, level).dag(),
                )
                for level in range(target.local_dimension)
            )
            for ordinal in range(len(self._target.model.subsystems))
        )

    def solver_metadata(self) -> dict[str, Any]:
        """Return normalized, public-safe solver facts for result metadata."""
        return {
            "solver": self._solver_used,
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
            self._solver_used = "mesolve"
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
        if abs(run.start_time) > TIME_EPSILON:
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
            self._solver_used = "propagator"
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
        if run.start_time < input_time - TIME_EPSILON:
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
            apply_ready_actions(pending_actions, start_time, frames)
            if not enabled[source_index]:
                continue
            for child, binding in zip(block.controls, block.control_bindings):
                pulses.append(self._bind_child(child, binding, start_time, frames))
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
            apply_actions(actions, frames)
        if run.end_time <= input_time + TIME_EPSILON:
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
            collapse_operators=(
                self._always_on_collapse_operators() + tuple(local_collapse)
            ),
            output_frames=frames,
        )

    def _always_on_collapse_operators(self) -> tuple[Any, ...]:
        """Expand resolved always-on terms only when dynamics need them."""
        if self._collapse_operators is None:
            self._collapse_operators = self._build_always_on_noise(
                self._always_on_noise
            )
        return self._collapse_operators

    def _frame_unitary(self, frames: dict[Any, float]) -> Any:
        """Build the full-model basis transform for terminal frame angles."""
        factors = []
        levels = np.arange(self._target.model.physical_dimension)
        for subsystem_id in self._target.model.subsystem_ids:
            angle = frames.get(self._target.model.frame(subsystem_id), 0.0)
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
                tuple(range(self._target.model.physical_dimension))
                for _ in step.measured_indices
            )
            confusions = step.confusions or (None,) * len(step.measured_indices)
            for engine_index, classical_index, digit_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                maps,
                confusions,
            ):
                outcome = self._measure(engine_index, context)
                reported = digit_map[outcome]
                if confusion is not None:
                    reported = int(
                        context.rng.choice(len(confusion), p=confusion[:, reported])
                    )
                context.classical_memory[classical_index] = reported
            return
        if _condition_matches(step.condition, context.classical_memory):
            for engine_index in step.reset_indices:
                self._reset(engine_index, context)

    def finish_shot(self, context: _ShotContext) -> _PulseShotOutcome:
        """Return a NumPy copy; no solver value crosses the engine boundary."""
        return _PulseShotOutcome(
            final_state=(
                np.array(context.state.full(), dtype=complex, copy=True)
                if self._retain_final_state
                else None
            ),
            final_state_kind="density_matrix",
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
        identity = qeye(self._target.model.physical_dimension)
        subsystems = {
            subsystem.id: subsystem for subsystem in self._target.model.subsystems
        }
        for engine_index, device_operand in enumerate(
            self._engine_allocation.device_operands
        ):
            subsystem = subsystems[device_operand]
            local = (
                angular_rate_from_ghz(subsystem.anharmonicity_ghz)
                * self._local_number
                * (self._local_number - identity)
                / 2
            )
            drift.add_drift(local, engine_index)
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
            for ordinal in term.engine_indices
        ]

    def _validate_noise_ordinal(self, ordinal: int) -> int:
        """Validate and return one physical-model subsystem ordinal."""
        if type(ordinal) is not int or not 0 <= ordinal < len(
            self._target.model.subsystems
        ):
            raise BackendValidationError(
                f"unknown physical-model subsystem ordinal {ordinal!r}"
            )
        return ordinal

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
        child: PulseControl,
        binding: _PreparedControlBinding,
        block_start_time: float,
        frames: dict[Any, float],
    ) -> Pulse:
        """Convert one sampled control to an absolute-time qutip-qip pulse.

        Drive controls bind complex envelopes to X/Y quadratures and rotate by
        the negative accumulated virtual-frame angle. Detuning binds the local
        number operator; exchange binds the two-qutrit ladder interaction on
        its declared coupling edge.
        """
        absolute_tlist = (
            block_start_time + child.start_offset + np.asarray(child.waveform.times)
        )
        coefficients = np.asarray(child.waveform.values)
        if not binding.engine_indices:
            raise BackendValidationError("transmon controls require engine indices")
        # Detuning and exchange envelopes are real: `PulseBlock` construction
        # already rejected a complex one through target validation, so taking `.real`
        # here discards nothing and needs no second check.
        if binding.kind == "detuning":
            if len(binding.engine_indices) != 1:
                raise BackendValidationError("detuning requires one engine index")
            return Pulse(
                self._local_number,
                binding.engine_indices[0],
                tlist=absolute_tlist,
                coeff=coefficients.real,
                spline_kind="cubic",
                label="detuning",
            )
        if binding.kind == "exchange":
            if len(binding.engine_indices) != 2:
                raise BackendValidationError("exchange requires two engine indices")
            targets = list(binding.engine_indices)
            exchange = tensor(
                self._local_annihilation.dag(), self._local_annihilation
            ) + tensor(self._local_annihilation, self._local_annihilation.dag())
            return Pulse(
                exchange,
                targets,
                tlist=absolute_tlist,
                coeff=coefficients.real,
                spline_kind="cubic",
                label="exchange",
            )
        if binding.kind != "drive":
            raise BackendValidationError("unknown transmon control kind")
        if len(binding.engine_indices) != 1:
            raise BackendValidationError("drive requires one engine index")
        engine_index = binding.engine_indices[0]
        subsystem_id = self._engine_allocation.device_operands[engine_index]
        phase = np.exp(
            -1j
            * frames.get(
                self._target.model.frame(subsystem_id),
                0.0,
            )
        )
        envelope = phase * coefficients
        x_operator = self._local_annihilation + self._local_annihilation.dag()
        y_operator = -1j * (self._local_annihilation - self._local_annihilation.dag())
        pulse = Pulse(
            x_operator,
            engine_index,
            tlist=absolute_tlist,
            coeff=envelope.real,
            spline_kind="cubic",
            label="drive",
        )
        pulse.add_coherent_noise(
            y_operator,
            engine_index,
            tlist=absolute_tlist,
            coeff=envelope.imag,
        )
        return pulse
