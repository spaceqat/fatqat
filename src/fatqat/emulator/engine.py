"""Private whole-plan orchestration for superconducting pulse execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np

from ..backends._execution_analysis import (
    _OperationExecutionFacts,
    _analyze_terminal_measurements,
)
from ..backends.steps import MeasurementStep, ResetStep
from ..errors import BackendValidationError
from .scheduling import (
    SchedulingMode,
    _ScheduledPulseRun,
    _validate_schedule_mode,
    schedule_pulse_run,
)
from .planning import PulsePlanStep
from .pulse import PulseBlock
from .superconducting import FrameRef


@dataclass
class _ShotContext:
    """Engine-owned mutable state that survives every run and boundary."""

    state: Any
    classical_memory: list[int]
    rng: np.random.Generator
    frame_angles: dict[FrameRef, float] = field(default_factory=dict)
    time: float = 0.0


class _PulseModelRunner(Protocol):
    """Internal solver/boundary contract consumed by :class:`PulseEngine`.

    The engine owns plan traversal, shot strategy, scheduling, conditions, and
    boundary placement. A runner owns model-specific state evolution,
    measurement/reset physics, and conversion of one completed shot into a
    backend-facing payload. The protocol deliberately uses ``Any`` so QuTiP
    values remain confined to the concrete adapter.
    """

    def initial_state(self) -> Any: ...

    def copy_state(self, state: Any) -> Any: ...

    def evolve(
        self,
        run: _ScheduledPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None: ...

    def propagator(
        self, run: _ScheduledPulseRun, *, apply_final_frame: bool = True
    ) -> Any: ...

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None: ...

    def finish_shot(self, context: _ShotContext) -> Any: ...


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, classical_memory: list[int]
) -> bool:
    """Return whether every lowered classical equality term is satisfied."""
    return condition is None or all(
        classical_memory[index] == value for index, value in condition
    )


class PulseEngine:
    """Internal whole-plan orchestrator for pulse execution.

    ``PulseEngine`` is the pulse counterpart of the NumPy simulator engine's
    orchestration layer. It selects the shared fast versus per-shot strategy,
    divides a plan at measurement/reset boundaries, schedules each continuous
    pulse region, evaluates classical conditions, and delegates physical work
    to a model runner. It does not construct Hamiltonians or know QuTiP.

    This class is private implementation machinery. Applications call
    :meth:`fatqat.backends.PulseBackend.run` or
    :meth:`fatqat.backends.PulseBackend.propagator` instead.
    """

    def __init__(
        self, runner: _PulseModelRunner, *, schedule_mode: SchedulingMode = "ASAP"
    ) -> None:
        self._runner = runner
        self._schedule_mode = _validate_schedule_mode(schedule_mode)

    def run(
        self,
        plan: Iterable[PulsePlanStep],
        *,
        shots: int,
        n_clbits: int,
        rng: np.random.Generator,
    ) -> tuple[Any, ...]:
        """Execute a lowered plan and return one runner payload per shot.

        Static plans evolve once and replay only terminal measurement. Dynamic
        plans (conditions, resets, or subsystem reuse after measurement) replay
        the complete trajectory for each shot. The supplied generator is
        shared in deterministic shot order.
        """
        if type(shots) is not int or shots <= 0:
            raise BackendValidationError("pulse engine shots must be a positive int")
        if type(n_clbits) is not int or n_clbits < 0:
            raise BackendValidationError(
                "pulse engine classical width must be a non-negative int"
            )
        frozen_plan = tuple(plan)
        is_dynamic, terminal_measurements = self._analyze_plan(frozen_plan)
        if is_dynamic:
            return self._run_per_shot(
                frozen_plan, shots=shots, n_clbits=n_clbits, rng=rng
            )
        return self._run_fast(
            frozen_plan,
            terminal_measurements,
            shots=shots,
            n_clbits=n_clbits,
            rng=rng,
        )

    def propagator(
        self,
        plan: Iterable[PulsePlanStep],
        *,
        apply_final_frame: bool = True,
    ) -> Any:
        """Schedule and propagate one coherent boundary-free pulse plan.

        Measurement, reset, classical conditions, unknown step types, and an
        empty plan are rejected before delegation to the model runner.
        """
        blocks = []
        for step in plan:
            if isinstance(step, MeasurementStep):
                raise BackendValidationError("propagator does not support measurement")
            if isinstance(step, ResetStep):
                raise BackendValidationError("propagator does not support reset")
            if not isinstance(step, PulseBlock):
                raise BackendValidationError(
                    "pulse plan contains an unknown execution step"
                )
            if step.condition is not None:
                raise BackendValidationError(
                    "propagator does not support classically conditioned operations"
                )
            blocks.append(step)
        if not blocks:
            raise BackendValidationError("cannot propagate an empty pulse plan")
        run = schedule_pulse_run(
            blocks,
            boundary_time=0.0,
            mode=self._schedule_mode,
        )
        return self._runner.propagator(run, apply_final_frame=apply_final_frame)

    def _run_per_shot(
        self,
        plan: tuple[PulsePlanStep, ...],
        *,
        shots: int,
        n_clbits: int,
        rng: np.random.Generator,
    ) -> tuple[Any, ...]:
        """Replay one complete dynamic trajectory for every requested shot."""
        outcomes = []
        for _ in range(shots):
            context = _ShotContext(
                state=self._runner.initial_state(),
                classical_memory=[0] * n_clbits,
                rng=rng,
            )
            self._run_one_shot(plan, context)
            outcomes.append(self._runner.finish_shot(context))
        return tuple(outcomes)

    @staticmethod
    def _analyze_plan(
        plan: tuple[PulsePlanStep, ...],
    ) -> tuple[bool, tuple[MeasurementStep, ...]]:
        """Return the shared terminal-measurement fast-path decision."""
        is_dynamic, measurements = _analyze_terminal_measurements(
            plan, PulseEngine._operation_execution_facts
        )
        return is_dynamic, measurements

    @staticmethod
    def _operation_execution_facts(step: PulsePlanStep) -> _OperationExecutionFacts:
        """Describe one pulse operation for shared dynamic-plan analysis."""
        if isinstance(step, PulseBlock):
            return _OperationExecutionFacts(
                target_indices=step.target_indices,
                is_conditioned=step.condition is not None,
            )
        if isinstance(step, ResetStep):
            return _OperationExecutionFacts(
                target_indices=step.reset_indices,
                is_conditioned=step.condition is not None,
            )
        raise BackendValidationError("pulse plan contains an unknown execution step")

    def _run_fast(
        self,
        plan: tuple[PulsePlanStep, ...],
        terminal_measurements: tuple[MeasurementStep, ...],
        *,
        shots: int,
        n_clbits: int,
        rng: np.random.Generator,
    ) -> tuple[Any, ...]:
        """Evolve a static plan once, then sample its terminal measurements."""
        context = _ShotContext(
            state=self._runner.initial_state(),
            classical_memory=[0] * n_clbits,
            rng=rng,
        )
        self._run_one_shot(plan, context, defer_measurements=True)

        source_state = (
            self._runner.copy_state(context.state)
            if terminal_measurements and shots > 1
            else context.state
        )
        outcomes = []
        for shot in range(shots):
            sample_context = (
                context
                if shot == 0
                else _ShotContext(
                    state=self._runner.copy_state(source_state),
                    classical_memory=[0] * n_clbits,
                    rng=rng,
                    frame_angles=dict(context.frame_angles),
                    time=context.time,
                )
            )
            for step in terminal_measurements:
                self._runner.execute_boundary(step, sample_context)
            outcomes.append(self._runner.finish_shot(sample_context))
        return tuple(outcomes)

    def _run_one_shot(
        self,
        plan: tuple[PulsePlanStep, ...],
        context: _ShotContext,
        *,
        defer_measurements: bool = False,
    ) -> None:
        """Traverse one trajectory, flushing pulses at physical boundaries.

        Consecutive pulse blocks share one scheduled continuous region, so
        disjoint controls may evolve concurrently. Measurement and reset flush
        that region before the boundary is applied. With deferred measurement,
        terminal measurements are left for fast-path resampling.
        """
        pending: list[PulseBlock] = []

        def flush() -> None:
            if not pending:
                return
            run = schedule_pulse_run(
                pending,
                boundary_time=context.time,
                mode=self._schedule_mode,
            )
            enabled = tuple(
                _condition_matches(block.condition, context.classical_memory)
                for block in run.blocks
            )
            self._runner.evolve(run, context, enabled)
            context.time = run.end_time
            pending.clear()

        for step in plan:
            if isinstance(step, PulseBlock):
                # A condition reads the classical memory established by the
                # preceding boundary.  It cannot change within a continuous
                # pulse region, so guarded and unguarded blocks can be placed
                # together.  `evolve()` receives one enable flag per block.
                pending.append(step)
            elif isinstance(step, (MeasurementStep, ResetStep)):
                flush()
                if not (defer_measurements and isinstance(step, MeasurementStep)):
                    self._runner.execute_boundary(step, context)
            else:
                raise BackendValidationError(
                    "pulse plan contains an unknown execution step"
                )
        flush()
