"""Private QuTiP qutrit operators for a bound three-level atom target."""

from __future__ import annotations

from typing import Any

import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    basis,
    coefficient,
    ket2dm,
    mesolve,
    propagator as qutip_propagator,
    qeye,
    sesolve,
    tensor,
)
from qutip_qip.pulse import Pulse

from ..._backends.steps import MeasurementStep, ResetStep
from ..._index_allocation import _EngineAllocation
from ...errors import BackendValidationError
from ..._pulse_values import PulseControl
from .._core.adapter_common import (
    _BoundDynamics,
    _BoundFrames,
    apply_actions,
    apply_ready_actions,
)
from .._core.engine import _ShotContext, _condition_matches
from .._core.lindblad import ResolvedLindbladTerm
from .._core.outcome import ExecutionMode, _PulseShotOutcome
from .._core.pulse import PhaseShift, PhaseSwap
from .._core.scheduling import _ScheduledPulseRun
from .._core.value_validation import TIME_EPSILON
from .._qutip_boundaries import (
    _apply_qutip_reset,
    _expand_qutip_local,
    _sample_projective_qutip_state,
    _solve_one_qutip_trajectory,
)
from .._core.target import _PreparedControlBinding
from .._qutip_runtime import _qutip_runtime_details
from .target import _Atom3LevelTarget

# Use QuTiP's native method and error tolerances; only raise its work ceiling.
_SOLVER_OVERRIDES = {"nsteps": 100000}


class _Atom3LevelQutipAdapter:
    """Private atom-only qutrit operator owner; no values leave this module."""

    def __init__(
        self,
        target: _Atom3LevelTarget,
        *,
        engine_allocation: _EngineAllocation,
        background_noise: tuple[ResolvedLindbladTerm, ...] = (),
        execution_mode: ExecutionMode = "density_matrix",
        retain_final_state: bool = True,
    ) -> None:
        if not isinstance(target, _Atom3LevelTarget):
            raise BackendValidationError("atom adapter requires an atom target")
        if type(retain_final_state) is not bool:
            raise BackendValidationError("retain_final_state must be a bool")
        if execution_mode not in ("statevector", "density_matrix", "trajectory"):
            raise BackendValidationError(
                f"unknown three-level atom execution mode {execution_mode!r}"
            )
        self._target = target
        if (
            engine_allocation.device_operands != target.device_labels
            or engine_allocation.system_dims != (3,) * len(target.device_labels)
        ):
            raise BackendValidationError(
                "atom engine allocation must cover the complete target in target order"
            )
        self._engine_allocation = engine_allocation
        # QuTiP factor 0 is already FATQAT public subsystem 0; preserve the
        # allocation order directly rather than translating tensor factors.
        self._retain_final_state = retain_final_state
        self._execution_mode = execution_mode
        self._solvers_used: set[str] = set()
        self._background_noise = tuple(background_noise)
        self._collapse_operators: tuple[Any, ...] | None = None
        self._dims = list(engine_allocation.system_dims)
        self.local_raman_raising = Qobj(
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], complex)
        )
        self.local_rydberg_raising = Qobj(
            np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], complex)
        )
        self.local_rydberg_number = Qobj(np.diag([0, 0, 1]))
        self._projectors = tuple(
            tuple(
                _expand_qutip_local(self._dims, ordinal, ket2dm(basis(3, level)))
                for level in range(3)
            )
            for ordinal in range(engine_allocation.n_subsystems)
        )
        self._reset_operators = tuple(
            tuple(
                _expand_qutip_local(
                    self._dims,
                    ordinal,
                    basis(3, 0) * basis(3, level).dag(),
                )
                for level in range(3)
            )
            for ordinal in range(engine_allocation.n_subsystems)
        )

    def runtime_details(self) -> dict[str, Any]:
        """Return public, normalized numerical integration facts."""
        return _qutip_runtime_details(self._solvers_used, _SOLVER_OVERRIDES)

    @staticmethod
    def raman_frame_multiplier(theta: float) -> complex:
        return np.exp(-1j * theta)

    @staticmethod
    def rydberg_frame_multiplier(theta: float) -> complex:
        return np.exp(1j * theta)

    @staticmethod
    def local_frame(theta: float) -> Qobj:
        return Qobj(np.diag([1.0, np.exp(1j * theta), 1.0]))

    def initial_state(self) -> Any:
        ket = tensor(*tuple(basis(dim, 0) for dim in self._dims))
        return ket2dm(ket) if self._execution_mode == "density_matrix" else ket

    @staticmethod
    def copy_state(state: Any) -> Any:
        return state.copy()

    def _validate_state(self, state: Any) -> None:
        if not isinstance(state, Qobj):
            raise TypeError("three-level atom runner requires a QuTiP state")
        if self._execution_mode == "density_matrix" and not state.isoper:
            raise TypeError(
                "three-level atom density-matrix runner requires an operator"
            )
        if self._execution_mode != "density_matrix" and not state.isket:
            raise TypeError("three-level atom statevector runner requires a ket")

    def interaction_drift(self) -> Qobj:
        drift = 0 * tensor(*tuple(qeye(dim) for dim in self._dims))
        for value in self._target.interactions:
            factors = [qeye(dim) for dim in self._dims]
            factors[self._engine_allocation.engine_index(value.first)] = (
                self.local_rydberg_number
            )
            factors[self._engine_allocation.engine_index(value.second)] = (
                self.local_rydberg_number
            )
            drift += value.signed_strength_rad_per_us * tensor(*factors)
        return drift

    def _bind_child(
        self,
        child: PulseControl,
        binding: _PreparedControlBinding,
        block_start_time: float,
        frames: dict[Any, float],
    ) -> Pulse:
        """Bind one full complex Rabi envelope as atom qutrit X/Y quadratures.

        The factor one-half belongs solely to this Hamiltonian assembly:
        ``H = (Omega sigma_+ + Omega* sigma_-)/2``.  Realization rules supply
        the full calibrated Rabi rate and must not compensate for it.
        """
        if len(binding.engine_indices) != 1:
            raise BackendValidationError("atom controls require one engine index")
        engine_index = binding.engine_indices[0]
        qutip_target = engine_index
        coefficients = np.asarray(child.waveform.values, dtype=complex)
        site = self._engine_allocation.device_operands[engine_index]
        frame_angle = frames.get(self._target.model.frame(site), 0.0)
        if binding.kind == "raman_01":
            envelope = self.raman_frame_multiplier(frame_angle) * coefficients
            lowering = self.local_raman_raising.dag()
            label = "raman_01"
        elif binding.kind == "rydberg_1r":
            envelope = self.rydberg_frame_multiplier(frame_angle) * coefficients
            lowering = self.local_rydberg_raising.dag()
            label = "rydberg_1r"
        else:  # Target binding owns the definitive foreign-address check.
            raise BackendValidationError("atom pulse control has an unknown transition")

        absolute_tlist = (
            block_start_time + child.start_offset + np.asarray(child.waveform.times)
        )
        x_operator = lowering + lowering.dag()
        y_operator = -1j * (lowering - lowering.dag())
        pulse = Pulse(
            x_operator,
            qutip_target,
            tlist=absolute_tlist,
            coeff=0.5 * envelope.real,
            spline_kind="cubic",
            label=label,
        )
        pulse.add_coherent_noise(
            y_operator,
            qutip_target,
            tlist=absolute_tlist,
            coeff=0.5 * envelope.imag,
        )
        return pulse

    def evolve(
        self, run: _ScheduledPulseRun, context: _ShotContext, enabled: tuple[bool, ...]
    ) -> None:
        bound = self._bind_run(
            run,
            enabled=enabled,
            input_time=context.time,
            input_frames=context.frame_angles,
        )
        self._validate_state(context.state)
        solver_state = context.state
        if isinstance(bound, _BoundDynamics):
            collapse_operators = (
                self._background_collapse_operators()
                + self._bind_block_collapse_operators(run, enabled)
            )
            if self._execution_mode == "density_matrix":
                self._solvers_used.add("mesolve")
                result = mesolve(
                    bound.hamiltonian,
                    solver_state,
                    [context.time, run.end_time],
                    c_ops=collapse_operators,
                    options=_SOLVER_OVERRIDES,
                )
                solver_state = result.states[-1]
            elif collapse_operators:
                if self._execution_mode != "trajectory":
                    raise BackendValidationError(
                        "three-level atom statevector execution requires coherent "
                        "dynamics"
                    )
                self._solvers_used.add("mcsolve")
                solver_state = _solve_one_qutip_trajectory(
                    bound.hamiltonian,
                    collapse_operators,
                    solver_state,
                    start_time=context.time,
                    end_time=run.end_time,
                    rng=context.rng,
                    solver_options=_SOLVER_OVERRIDES,
                )
            else:
                self._solvers_used.add("sesolve")
                result = sesolve(
                    bound.hamiltonian,
                    solver_state,
                    [context.time, run.end_time],
                    options=_SOLVER_OVERRIDES,
                )
                solver_state = result.states[-1]
        context.state = solver_state
        context.frame_angles.clear()
        context.frame_angles.update(bound.output_frames)

    def propagator(
        self, run: _ScheduledPulseRun, *, apply_final_frame: bool = True
    ) -> Any:
        if abs(run.start_time) > TIME_EPSILON:
            raise BackendValidationError("a propagator run must start at time zero")
        bound = self._bind_run(
            run,
            enabled=(True,) * len(run.blocks),
            input_time=0.0,
            input_frames={},
        )
        if isinstance(bound, _BoundFrames):
            unitary = tensor(*tuple(qeye(dim) for dim in self._dims))
        else:
            self._solvers_used.add("propagator")
            unitary = qutip_propagator(
                bound.hamiltonian,
                run.end_time,
                options=_SOLVER_OVERRIDES,
            )
        return (
            self._frame_unitary(bound.output_frames) * unitary
            if apply_final_frame
            else unitary
        )

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None:
        """Measure qutrits into binary atom digits or conditionally reset them."""
        if isinstance(step, MeasurementStep):
            maps = step.reported_digit_maps or ((0, 1, 1),) * len(step.measured_indices)
            confusions = step.confusions or (None,) * len(step.measured_indices)
            for engine_index, classical_index, digit_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                maps,
                confusions,
            ):
                outcome = self._measure(engine_index, context)
                try:
                    reported = digit_map[outcome]
                except IndexError as exc:
                    raise BackendValidationError(
                        "atom measurement digit map must cover all qutrit outcomes"
                    ) from exc
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
        """Convert the solver component to a copied NumPy payload."""
        self._validate_state(context.state)
        final_state = None
        if self._retain_final_state:
            array = np.asarray(context.state.full(), dtype=complex)
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

    def _measure(self, ordinal: int, context: _ShotContext) -> int:
        """Sample and collapse one physical qutrit in the active representation."""
        try:
            projectors = self._projectors[ordinal]
        except (IndexError, TypeError):
            raise BackendValidationError(f"unknown atom site {ordinal!r}") from None
        self._validate_state(context.state)
        outcome, context.state = _sample_projective_qutip_state(
            context.state,
            projectors,
            context.rng,
        )
        return outcome

    def _reset(self, ordinal: int, context: _ShotContext) -> None:
        """Apply the deterministic local qutrit-to-ground reset channel."""
        try:
            operators = self._reset_operators[ordinal]
        except (IndexError, TypeError):
            raise BackendValidationError(f"unknown atom site {ordinal!r}") from None
        self._validate_state(context.state)
        context.state = _apply_qutip_reset(context.state, operators, context.rng)

    def _background_collapse_operators(self) -> tuple[Any, ...]:
        if self._collapse_operators is None:
            self._collapse_operators = self._expand_collapse_terms(
                self._background_noise
            )
        return self._collapse_operators

    def _expand_collapse_terms(
        self, terms: tuple[ResolvedLindbladTerm, ...]
    ) -> tuple[Any, ...]:
        result: list[Any] = []
        for term in terms:
            local = Qobj(term.local_operator)
            for engine_index in term.engine_indices:
                if not 0 <= engine_index < len(self._dims):
                    raise BackendValidationError(
                        f"unknown atom Lindblad engine index {engine_index!r}"
                    )
                result.append(_expand_qutip_local(self._dims, engine_index, local))
        return tuple(result)

    def _bind_block_collapse_operators(
        self,
        run: _ScheduledPulseRun,
        enabled: tuple[bool, ...],
    ) -> tuple[Any, ...]:
        result: list[Any] = []
        for block, start, is_enabled in zip(run.blocks, run.starts, enabled):
            if not is_enabled or block.duration == 0.0:
                continue
            end = start + block.duration

            def window(
                time: float,
                _args: dict[str, Any] | None = None,
                *,
                start_time: float = start,
                end_time: float = end,
            ) -> float:
                return float(start_time <= time < end_time)

            for operator in self._expand_collapse_terms(block.noise):
                result.append(QobjEvo([operator, coefficient(window, args={})]))
        return tuple(result)

    def _bind_run(
        self,
        run: _ScheduledPulseRun,
        *,
        enabled: tuple[bool, ...],
        input_time: float,
        input_frames: dict[Any, float],
    ) -> _BoundFrames | _BoundDynamics:
        """Bind controls at absolute time, including all occupied-pair drift."""
        if len(enabled) != len(run.blocks):
            raise BackendValidationError(
                "pulse enable flags must align with the placed run"
            )
        if run.start_time < input_time - TIME_EPSILON:
            raise BackendValidationError("placed pulse runs must be time ordered")

        frames = dict(input_frames)
        pulses: list[Pulse] = []
        pending_actions: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]] = (
            []
        )
        for source_index in sorted(
            range(len(run.blocks)), key=lambda index: (run.starts[index], index)
        ):
            block = run.blocks[source_index]
            start_time = run.starts[source_index]
            apply_ready_actions(pending_actions, start_time, frames)
            if not enabled[source_index]:
                continue
            pulses.extend(
                self._bind_child(child, binding, start_time, frames)
                for child, binding in zip(block.controls, block.control_bindings)
            )
            pending_actions.append(
                (start_time + block.duration, source_index, block.post_actions)
            )

        for _end, _source, actions in sorted(
            pending_actions, key=lambda event: event[:2]
        ):
            apply_actions(actions, frames)
        if run.end_time <= input_time + TIME_EPSILON:
            return _BoundFrames(output_frames=frames)

        hamiltonian = self.interaction_drift()
        for pulse in pulses:
            contribution, collapse = pulse.get_noisy_qobjevo(self._dims)
            if collapse:
                raise BackendValidationError(
                    "atom coherent pulse binding produced collapse terms"
                )
            hamiltonian += contribution
        return _BoundDynamics(hamiltonian=hamiltonian, output_frames=frames)

    def _frame_unitary(self, frames: dict[Any, float]) -> Qobj:
        public_factors = tuple(
            self.local_frame(frames.get(self._target.model.frame(site), 0.0))
            for site in self._engine_allocation.device_operands
        )
        return tensor(*public_factors)
