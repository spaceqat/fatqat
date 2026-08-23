"""Private QuTiP execution adapter for two-level atoms."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    basis,
    coefficient,
    ket2dm,
    mcsolve,
    mesolve,
    propagator,
    qeye,
    sesolve,
    tensor,
)
from scipy.sparse import diags

from ..._backends.steps import MeasurementStep, ResetStep
from ..._index_allocation import _EngineAllocation
from ...errors import BackendValidationError
from ..._pulse_values import PulseControl
from .._core.engine import _ShotContext
from .._core.lindblad import ResolvedLindbladTerm
from .._core.outcome import ExecutionMode, _PulseShotOutcome
from .._core.qutip_allocation import _QutipEngineAllocation
from .._core.scheduling import _ScheduledPulseRun
from .._core.value_validation import TIME_EPSILON
from .._core.waveform import _REQUESTED_SPLINE_DEGREE
from .target import _Atom2LevelTarget

_SOLVER_OPTIONS = {
    "method": "vern9",
    "atol": 1e-11,
    "rtol": 1e-9,
    "nsteps": 100000,
}


def _seed_entropies(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(getattr(seed, "entropy", seed)) for seed in values)


class _Atom2LevelQutipAdapter:
    """Ket-preserving coherent runner for one bound two-level target."""

    def __init__(
        self,
        target: _Atom2LevelTarget,
        *,
        engine_allocation: _EngineAllocation,
        background_noise: tuple[ResolvedLindbladTerm, ...] = (),
        execution_mode: ExecutionMode = "statevector",
        retain_final_state: bool = True,
    ) -> None:
        if not isinstance(target, _Atom2LevelTarget):
            raise BackendValidationError(
                "two-level adapter requires a two-level target"
            )
        if type(retain_final_state) is not bool:
            raise BackendValidationError("retain_final_state must be a bool")
        if execution_mode not in ("statevector", "density_matrix", "trajectory"):
            raise BackendValidationError(
                f"unknown two-level execution mode {execution_mode!r}"
            )
        if (
            engine_allocation.device_operands != target.device_labels
            or engine_allocation.system_dims != (2,) * len(target.device_labels)
        ):
            raise BackendValidationError(
                "two-level engine allocation must cover the complete target in "
                "target order"
            )

        self._target = target
        self._engine_allocation = engine_allocation
        self._qutip_allocation = _QutipEngineAllocation(engine_allocation)
        self._retain_final_state = retain_final_state
        self._execution_mode = execution_mode
        self._solver_used = "none"
        self._site_count = engine_allocation.n_subsystems
        self._dims = list(self._qutip_allocation.qutip_dims)
        self.local_raising = Qobj(np.asarray([[0.0, 0.0], [1.0, 0.0]], complex))
        self.local_number = Qobj(np.diag([0.0, 1.0]))
        identity = tensor(*(qeye(2) for _ in range(self._site_count)))
        self._global_raising = sum(
            (
                self._expand_local(site, self.local_raising)
                for site in range(self._site_count)
            ),
            0 * identity,
        )
        self._global_number = sum(
            (
                self._expand_local(site, self.local_number)
                for site in range(self._site_count)
            ),
            0 * identity,
        )
        self._interaction_drift = self._build_interaction_drift()
        self._collapse_operators = self._build_collapse_operators(background_noise)

    def solver_metadata(self) -> dict[str, Any]:
        return {
            "solver": self._solver_used,
            "options": dict(_SOLVER_OPTIONS),
        }

    def initial_state(self) -> Any:
        ket = tensor(*(basis(2, 0) for _ in range(self._site_count)))
        return ket2dm(ket) if self._execution_mode == "density_matrix" else ket

    def copy_state(self, state: Any) -> Any:
        self._validate_state(state)
        return state.copy()

    def _validate_state(self, state: Any) -> None:
        """Validate the active solver representation without copying it."""
        if not isinstance(state, Qobj):
            raise TypeError("two-level runner requires a QuTiP state")
        if self._execution_mode == "density_matrix" and not state.isoper:
            raise TypeError("two-level density-matrix runner requires a QuTiP operator")
        if self._execution_mode != "density_matrix" and not state.isket:
            raise TypeError("two-level statevector runner requires a QuTiP ket")

    @property
    def interaction_drift(self) -> Any:
        return self._interaction_drift

    def evolve(
        self,
        run: _ScheduledPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None:
        self._validate_state(context.state)
        hamiltonian = self._bind_run(
            run,
            enabled=enabled,
            input_time=context.time,
        )
        if self._execution_mode == "density_matrix":
            self._solver_used = "mesolve"
            result = mesolve(
                hamiltonian,
                context.state,
                [context.time, run.end_time],
                c_ops=(
                    self._collapse_operators
                    + self._bind_block_collapse_operators(run, enabled)
                ),
                options=_SOLVER_OPTIONS,
            )
        elif self._execution_mode == "statevector":
            self._solver_used = "sesolve"
            result = sesolve(
                hamiltonian,
                context.state,
                [context.time, run.end_time],
                options=_SOLVER_OPTIONS,
            )
        else:
            raise BackendValidationError(
                "trajectory evolution requires terminal batch execution"
            )
        context.state = result.states[-1]

    def propagator(
        self, run: _ScheduledPulseRun, *, apply_final_frame: bool = True
    ) -> Any:
        if type(apply_final_frame) is not bool:
            raise BackendValidationError("apply_final_frame must be a bool")
        if abs(run.start_time) > TIME_EPSILON:
            raise BackendValidationError("a propagator run must start at time zero")
        hamiltonian = self._bind_run(
            run,
            enabled=(True,) * len(run.blocks),
            input_time=0.0,
        )
        self._solver_used = "propagator"
        return propagator(
            hamiltonian,
            run.end_time,
            options=_SOLVER_OPTIONS,
        )

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None:
        if isinstance(step, ResetStep):
            raise BackendValidationError("two-level execution does not support reset")
        if self._execution_mode == "density_matrix":
            raise BackendValidationError(
                "two-level density-matrix execution has no measurement boundary"
            )
        maps = step.reported_digit_maps or ((0, 1),) * len(step.measured_indices)
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
            except IndexError as error:
                raise BackendValidationError(
                    "two-level measurement digit map must cover both physical outcomes"
                ) from error
            if confusion is not None:
                reported = int(
                    context.rng.choice(len(confusion), p=confusion[:, reported])
                )
            context.classical_memory[classical_index] = reported

    def finish_shot(self, context: _ShotContext) -> _PulseShotOutcome:
        final_state = None
        if self._retain_final_state:
            state = self._state_array(context.state)
            final_state = np.array(state, dtype=complex, copy=True)
        return _PulseShotOutcome(
            final_state=final_state,
            final_state_kind=(
                "density_matrix"
                if self._execution_mode == "density_matrix"
                else "statevector"
            ),
            classical_digits=tuple(context.classical_memory),
        )

    def run_trajectory_batch(
        self,
        scheduled_run: _ScheduledPulseRun,
        *,
        ntraj: int,
        seeds: tuple[int, ...],
    ) -> tuple[Any, ...]:
        if self._execution_mode != "trajectory":
            raise BackendValidationError(
                "two-level runner is not configured for trajectory execution"
            )
        if type(ntraj) is not int or ntraj <= 0:
            raise BackendValidationError("trajectory count must be a positive int")
        if (
            not isinstance(seeds, tuple)
            or len(seeds) != ntraj
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise BackendValidationError(
                "trajectory seeds must be one non-negative int per trajectory"
            )
        hamiltonian = self._bind_run(
            scheduled_run,
            enabled=(True,) * len(scheduled_run.blocks),
            input_time=scheduled_run.start_time,
        )
        collapse_operators = (
            self._collapse_operators
            + self._bind_block_collapse_operators(
                scheduled_run,
                (True,) * len(scheduled_run.blocks),
            )
        )
        options = {
            **_SOLVER_OPTIONS,
            "store_final_state": True,
            "keep_runs_results": True,
            "progress_bar": False,
        }
        self._solver_used = "mcsolve"
        result = mcsolve(
            hamiltonian,
            self.initial_state(),
            [scheduled_run.start_time, scheduled_run.end_time],
            c_ops=collapse_operators,
            ntraj=ntraj,
            seeds=list(seeds),
            options=options,
        )
        final_states = result.runs_final_states
        if final_states is None:
            raise BackendValidationError(
                "mcsolve retained run-final states were unavailable"
            )
        if len(final_states) != ntraj:
            raise BackendValidationError(
                "mcsolve returned the wrong number of retained run-final states"
            )
        returned_seeds = getattr(result, "seeds", None)
        if not isinstance(returned_seeds, Iterable):
            raise BackendValidationError(
                "mcsolve returned retained trajectories without seed order"
            )
        normalized_returned_seeds = _seed_entropies(returned_seeds)
        if normalized_returned_seeds != seeds:
            raise BackendValidationError(
                "mcsolve returned retained trajectories in an unexpected seed order"
            )
        if any(
            not isinstance(state, Qobj) or not state.isket for state in final_states
        ):
            raise BackendValidationError(
                "mcsolve retained run-final states must all be kets"
            )
        return tuple(state.copy() for state in final_states)

    def _expand_local(self, canonical_axis: int, operator: Any) -> Any:
        factors = [qeye(2) for _ in range(self._site_count)]
        factors[self._qutip_allocation.factor_index(canonical_axis)] = operator
        return tensor(*factors)

    def _build_interaction_drift(self) -> Any:
        dimension = 2**self._site_count
        basis_indices = np.arange(dimension, dtype=np.int64)
        diagonal = np.zeros(dimension, dtype=float)
        for interaction in self._target.interactions:
            first = self._engine_allocation.engine_index(interaction.first)
            second = self._engine_allocation.engine_index(interaction.second)
            both_occupied = ((basis_indices >> first) & 1) * (
                (basis_indices >> second) & 1
            )
            diagonal += interaction.signed_strength_rad_per_us * both_occupied
        return Qobj(
            diags(diagonal, offsets=0, format="csr"),
            dims=[self._dims, self._dims],
        )

    def _build_collapse_operators(
        self, bindings: tuple[ResolvedLindbladTerm, ...]
    ) -> tuple[Any, ...]:
        result: list[Any] = []
        for binding in bindings:
            local = Qobj(binding.local_operator)
            for engine_index in binding.engine_indices:
                if not 0 <= engine_index < self._site_count:
                    raise BackendValidationError(
                        f"unknown two-level Lindblad engine index {engine_index!r}"
                    )
                result.append(self._expand_local(engine_index, local))
        return tuple(result)

    def _bind_block_collapse_operators(
        self,
        run: _ScheduledPulseRun,
        enabled: tuple[bool, ...],
    ) -> tuple[Any, ...]:
        """Expand enabled block-scoped terms over their scheduled windows."""
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

            for term in block.noise:
                local = Qobj(term.local_operator)
                for engine_index in term.engine_indices:
                    if not 0 <= engine_index < self._site_count:
                        raise BackendValidationError(
                            "unknown two-level Lindblad engine index "
                            f"{engine_index!r}"
                        )
                    result.append(
                        QobjEvo(
                            [
                                self._expand_local(engine_index, local),
                                coefficient(window, args={}),
                            ]
                        )
                    )
        return tuple(result)

    @staticmethod
    def _windowed_coefficient(
        child: PulseControl, block_start: float, *, conjugate: bool = False
    ) -> Any:
        samples = np.asarray(child.waveform.values, dtype=complex)
        if conjugate:
            samples = np.conjugate(samples)
        start = block_start + child.start_offset
        end = start + child.waveform.duration
        interpolation = coefficient(
            samples,
            tlist=start + np.asarray(child.waveform.times),
            order=_REQUESTED_SPLINE_DEGREE,
        )

        def window(time: float, _args: dict[str, Any] | None = None) -> float:
            return float(start <= time <= end)

        return interpolation * coefficient(window, args={})

    def _bind_run(
        self,
        run: _ScheduledPulseRun,
        *,
        enabled: tuple[bool, ...],
        input_time: float,
    ) -> Any:
        if len(enabled) != len(run.blocks):
            raise BackendValidationError(
                "pulse enable flags must align with the placed run"
            )
        if run.start_time < input_time - TIME_EPSILON:
            raise BackendValidationError("placed pulse runs must be time ordered")
        terms: list[Any] = [self._interaction_drift]
        for block, block_start, is_enabled in zip(run.blocks, run.starts, enabled):
            if not is_enabled:
                continue
            for child, binding in zip(block.controls, block.control_bindings):
                if binding.kind == "drive":
                    terms.append(
                        [
                            0.5 * self._global_raising,
                            self._windowed_coefficient(child, block_start),
                        ]
                    )
                    terms.append(
                        [
                            0.5 * self._global_raising.dag(),
                            self._windowed_coefficient(
                                child, block_start, conjugate=True
                            ),
                        ]
                    )
                elif binding.kind == "detuning":
                    terms.append(
                        [
                            -self._global_number,
                            self._windowed_coefficient(child, block_start),
                        ]
                    )
                else:
                    raise BackendValidationError("unknown two-level atom control kind")
        return QobjEvo(terms)

    def _measure(self, canonical_axis: int, context: _ShotContext) -> int:
        state = self._statevector(context.state)
        basis_indices = np.arange(len(state), dtype=np.int64)
        digits = (basis_indices >> canonical_axis) & 1
        probabilities = np.bincount(
            digits,
            weights=np.abs(state) ** 2,
            minlength=2,
        ).astype(float)
        total = float(np.sum(probabilities))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("two-level measurement produced invalid probabilities")
        probabilities /= total
        outcome = int(context.rng.choice(2, p=probabilities))
        collapsed = np.array(state, dtype=complex, copy=True)
        collapsed[digits != outcome] = 0.0
        collapsed /= np.sqrt(probabilities[outcome] * total)
        context.state = Qobj(collapsed, dims=[self._dims, [1] * self._site_count])
        return outcome

    def _statevector(self, state: Any) -> np.ndarray:
        if not isinstance(state, Qobj) or not state.isket:
            raise TypeError("two-level coherent runner requires a QuTiP ket")
        vector = np.asarray(state.full(), dtype=complex).reshape(-1)
        if vector.shape != (2**self._site_count,) or not np.all(np.isfinite(vector)):
            raise RuntimeError("two-level coherent runner produced an invalid ket")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            raise RuntimeError("two-level coherent runner produced an invalid ket")
        return vector / norm

    def _state_array(self, state: Any) -> np.ndarray:
        if self._execution_mode == "density_matrix":
            if not isinstance(state, Qobj) or not state.isoper:
                raise TypeError(
                    "two-level density-matrix runner requires a QuTiP operator"
                )
            matrix = np.asarray(state.full(), dtype=complex)
            expected = 2**self._site_count
            if matrix.shape != (expected, expected) or not np.all(np.isfinite(matrix)):
                raise RuntimeError(
                    "two-level density-matrix runner produced an invalid state"
                )
            return matrix
        return self._statevector(state)


__all__: list[str] = []
