"""Private qutip-qip binding for full-qutrit superconducting evolution."""

from __future__ import annotations

from typing import Any

import numpy as np
from qutip import (
    Qobj,
    basis,
    destroy,
    ket2dm,
    mesolve,
    num,
    propagator as qutip_propagator,
    qeye,
    sesolve,
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
from .._core.outcome import ExecutionMode, _PulseShotOutcome
from .._core.pulse import PhaseShift, PhaseSwap, PulseBlock
from .._core.scheduling import _ScheduledPulseRun
from .._core.target import _PreparedControlBinding
from .._core.value_validation import TIME_EPSILON
from .._qutip_boundaries import (
    _apply_qutip_reset,
    _expand_qutip_local,
    _sample_projective_qutip_state,
    _solve_one_qutip_trajectory,
)
from .._qutip_runtime import _QutipRuntime
from .model import angular_rate_from_ghz
from .target import _TransmonTarget


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
        background_noise: tuple[ResolvedLindbladTerm, ...] = (),
        execution_mode: ExecutionMode = "density_matrix",
        retain_final_state: bool = True,
        max_step: float | None = None,
    ) -> None:
        """Adapt one already-bound physical target to the QuTiP layer.

        The engine allocation fixes public physical-factor order.
        ``background_noise`` contains already resolved local collapse terms;
        the adapter never interprets source noise descriptors.
        """
        if not isinstance(target, _TransmonTarget):
            raise BackendValidationError("transmon adapter requires a transmon target")
        self._target = target
        if type(retain_final_state) is not bool:
            raise BackendValidationError("retain_final_state must be a bool")
        if execution_mode not in ("statevector", "density_matrix", "trajectory"):
            raise BackendValidationError(
                f"unknown transmon execution mode {execution_mode!r}"
            )
        self._retain_final_state = retain_final_state
        self._execution_mode = execution_mode
        self._runtime = _QutipRuntime(max_step=max_step)
        self._background_noise = tuple(background_noise)
        self._collapse_operators: tuple[Any, ...] | None = None
        expected_operands = tuple(self._target.device_labels)
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
        # QuTiP factor 0 is already FATQAT public subsystem 0; no order
        # translation or result-side permutation is required.
        self._dims = list(engine_allocation.system_dims)
        self._local_annihilation = destroy(target.local_dimension)
        self._local_number = num(target.local_dimension)
        self._annihilation = tuple(
            _expand_qutip_local(self._dims, ordinal, self._local_annihilation)
            for ordinal in range(len(self._target.device_labels))
        )
        self._number = tuple(
            _expand_qutip_local(self._dims, ordinal, self._local_number)
            for ordinal in range(len(self._target.device_labels))
        )
        self._drift = self._build_drift()
        self._projectors = tuple(
            tuple(
                _expand_qutip_local(
                    self._dims,
                    ordinal,
                    ket2dm(basis(target.local_dimension, level)),
                )
                for level in range(target.local_dimension)
            )
            for ordinal in range(len(self._target.device_labels))
        )
        self._reset_operators = tuple(
            tuple(
                _expand_qutip_local(
                    self._dims,
                    ordinal,
                    basis(target.local_dimension, 0)
                    * basis(target.local_dimension, level).dag(),
                )
                for level in range(target.local_dimension)
            )
            for ordinal in range(len(self._target.device_labels))
        )

    def runtime_details(self) -> dict[str, Any]:
        """Return normalized QuTiP details for result metadata."""
        return self._runtime.details()

    def initial_state(self) -> Any:
        """Create the full-model physical ground state."""
        ket = tensor(*[basis(dimension, 0) for dimension in self._dims])
        return ket2dm(ket) if self._execution_mode == "density_matrix" else ket

    @staticmethod
    def copy_state(state: Any) -> Any:
        """Copy a state for an independent terminal-measurement trajectory."""
        return state.copy()

    def _validate_state(self, state: Any) -> None:
        if not isinstance(state, Qobj):
            raise TypeError("transmon runner requires a QuTiP state")
        if self._execution_mode == "density_matrix" and not state.isoper:
            raise TypeError("transmon density-matrix runner requires an operator")
        if self._execution_mode != "density_matrix" and not state.isket:
            raise TypeError("transmon statevector runner requires a ket")

    def evolve(
        self,
        run: _ScheduledPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None:
        """Evolve one placed region and commit enabled post-frame actions.

        False enable flags suppress a block's controls, operation-scoped noise,
        and post-actions while elapsed region time and background dynamics
        remain active.
        """
        bound = self._bind_run(
            run,
            enabled=enabled,
            input_time=context.time,
            input_frames=context.frame_angles,
        )
        self._validate_state(context.state)
        state = context.state
        if isinstance(bound, _BoundDynamics):
            solver_options = self._runtime.options_for(run.blocks)
            if self._execution_mode == "density_matrix":
                self._runtime.record_solver("mesolve")
                result = mesolve(
                    bound.hamiltonian,
                    state,
                    [context.time, run.end_time],
                    c_ops=bound.collapse_operators,
                    options=solver_options,
                )
                state = result.states[-1]
            elif bound.collapse_operators:
                if self._execution_mode != "trajectory":
                    raise BackendValidationError(
                        "transmon statevector execution requires coherent dynamics"
                    )
                self._runtime.record_solver("mcsolve")
                state = _solve_one_qutip_trajectory(
                    bound.hamiltonian,
                    bound.collapse_operators,
                    state,
                    start_time=context.time,
                    end_time=run.end_time,
                    rng=context.rng,
                    solver_options=solver_options,
                )
            else:
                self._runtime.record_solver("sesolve")
                result = sesolve(
                    bound.hamiltonian,
                    state,
                    [context.time, run.end_time],
                    options=solver_options,
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
            unitary = tensor(*[qeye(dimension) for dimension in self._dims])
        elif bound.collapse_operators:
            raise BackendValidationError(
                "propagator is unavailable for dissipative pulse evolution"
            )
        else:
            self._runtime.record_solver("propagator")
            unitary = qutip_propagator(
                bound.hamiltonian,
                run.end_time,
                options=self._runtime.options_for(run.blocks),
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
                self._background_collapse_operators() + tuple(local_collapse)
            ),
            output_frames=frames,
        )

    def _background_collapse_operators(self) -> tuple[Any, ...]:
        """Expand resolved background terms only when dynamics need them."""
        if self._collapse_operators is None:
            self._collapse_operators = self._build_background_noise(
                self._background_noise
            )
        return self._collapse_operators

    def _frame_unitary(self, frames: dict[Any, float]) -> Any:
        """Build the full-model basis transform for terminal frame angles."""
        factors = []
        levels = np.arange(self._target.local_dimension)
        for subsystem_id in self._engine_allocation.device_operands:
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
                tuple(range(self._target.local_dimension))
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
        """Return the final state after composing its terminal virtual frame."""
        self._validate_state(context.state)
        final_state = None
        if self._retain_final_state:
            state = context.state
            if context.frame_angles:
                frame_unitary = self._frame_unitary(context.frame_angles)
                if self._execution_mode == "density_matrix":
                    state = frame_unitary * state * frame_unitary.dag()
                else:
                    state = frame_unitary * state
            array = np.asarray(state.full(), dtype=complex)
            if self._execution_mode != "density_matrix":
                array = array.reshape(-1)
            final_state = np.array(array, dtype=complex, copy=True)
        return _PulseShotOutcome(
            final_state=final_state,
            final_state_kind=(
                "density_matrix"
                if self._execution_mode == "density_matrix"
                else "statevector"
            ),
            classical_digits=tuple(context.classical_memory),
        )

    def _build_drift(self) -> Drift:
        """Build the rotating-frame local-anharmonicity drift.

        Declared coupling edges contribute no static exchange; exchange enters
        only through an explicitly driven control pulse.
        """
        drift = Drift()
        identity = qeye(self._target.local_dimension)
        for engine_index, device_operand in enumerate(
            self._engine_allocation.device_operands
        ):
            subsystem = self._target._subsystems[device_operand]
            local = (
                angular_rate_from_ghz(subsystem.anharmonicity_ghz)
                * self._local_number
                * (self._local_number - identity)
                / 2
            )
            drift.add_drift(
                local,
                engine_index,
            )
        return drift

    def _build_background_noise(
        self, bindings: tuple[ResolvedLindbladTerm, ...]
    ) -> tuple[Any, ...]:
        """Build constant collapse terms from resolved background bindings."""
        noise_pulse = Pulse(None, None)
        for binding in bindings:
            for local_qobj, factor_index in self._lindblad_ops(binding):
                noise_pulse.add_lindblad_noise(
                    local_qobj,
                    factor_index,
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
            for local_qobj, factor_index in self._lindblad_ops(binding):
                noise_pulse.add_lindblad_noise(
                    local_qobj, factor_index, tlist=tlist, coeff=window
                )
        return noise_pulse

    def _lindblad_ops(self, term: ResolvedLindbladTerm) -> list[tuple[Any, int]]:
        """Adapt one backend-neutral term to local operators and QuTiP factors."""
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
            self._target.device_labels
        ):
            raise BackendValidationError(
                f"unknown physical-model subsystem ordinal {ordinal!r}"
            )
        return ordinal

    def _measure(self, ordinal: int, context: _ShotContext) -> int:
        """Sample one physical qutrit projector and collapse the active state."""
        self._validate_state(context.state)
        outcome, context.state = _sample_projective_qutip_state(
            context.state,
            self._projectors[ordinal],
            context.rng,
        )
        return outcome

    def _reset(self, ordinal: int, context: _ShotContext) -> None:
        """Apply the exact channel or sample one pure-state Kraus branch."""
        self._validate_state(context.state)
        context.state = _apply_qutip_reset(
            context.state,
            self._reset_operators[ordinal],
            context.rng,
        )

    def _bind_child(
        self,
        child: PulseControl,
        binding: _PreparedControlBinding,
        block_start_time: float,
        frames: dict[Any, float],
    ) -> Pulse:
        """Convert one sampled control to an absolute-time qutip-qip pulse.

        Drive controls bind half of each full complex Rabi envelope to the X/Y
        quadratures and rotate by the negative accumulated virtual-frame angle.
        Detuning binds the local number operator; exchange binds the two-qutrit
        ladder interaction on its declared coupling edge.
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
        factor_index = engine_index
        subsystem_id = self._engine_allocation.device_operands[engine_index]
        phase = np.exp(
            -1j
            * frames.get(
                self._target.model.frame(subsystem_id),
                0.0,
            )
        )
        envelope = 0.5 * phase * coefficients
        x_operator = self._local_annihilation + self._local_annihilation.dag()
        y_operator = -1j * (self._local_annihilation - self._local_annihilation.dag())
        pulse = Pulse(
            x_operator,
            factor_index,
            tlist=absolute_tlist,
            coeff=envelope.real,
            spline_kind="cubic",
            label="drive",
        )
        pulse.add_coherent_noise(
            y_operator,
            factor_index,
            tlist=absolute_tlist,
            coeff=envelope.imag,
        )
        return pulse
