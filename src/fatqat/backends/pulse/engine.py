"""Private whole-plan orchestration for superconducting pulse execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np

from ...backends.steps import MeasurementStep, ResetStep
from ...errors import BackendValidationError
from .execution import PlacementMode, _PlacedPulseRun, place_pulse_run
from .planning import PulsePlanStep
from .resolved import PulseBlock
from .superconducting import FrameRef


@dataclass
class _ShotContext:
    """Engine-owned mutable state that survives every run and boundary."""

    state: Any
    classical_memory: list[int]
    rng: np.random.Generator
    frame_angles: dict[FrameRef, float] = field(default_factory=dict)
    time_ns: float = 0.0


class _PulseModelRunner(Protocol):
    """Solver/boundary hooks invoked only by the whole-plan engine."""

    def initial_state(self) -> Any: ...

    def evolve(
        self,
        run: _PlacedPulseRun,
        context: _ShotContext,
        enabled: tuple[bool, ...],
    ) -> None: ...

    def execute_boundary(
        self, step: MeasurementStep | ResetStep, context: _ShotContext
    ) -> None: ...

    def finish_shot(self, context: _ShotContext) -> Any: ...


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, classical_memory: list[int]
) -> bool:
    return condition is None or all(
        classical_memory[index] == value for index, value in condition
    )


class PulseEngine:
    """Place and replay a complete lowered plan in deterministic shot order."""

    def __init__(
        self, runner: _PulseModelRunner, *, placement_mode: PlacementMode = "ASAP"
    ) -> None:
        if placement_mode not in ("ASAP", "ALAP"):
            raise BackendValidationError(
                "pulse placement mode must be 'ASAP' or 'ALAP'"
            )
        self._runner = runner
        self._placement_mode = placement_mode

    def execute(
        self,
        plan: Iterable[PulsePlanStep],
        *,
        shots: int,
        n_clbits: int,
        rng: np.random.Generator,
    ) -> tuple[Any, ...]:
        """Execute all shots and return one private runner payload per shot."""
        if type(shots) is not int or shots <= 0:
            raise BackendValidationError("pulse engine shots must be a positive int")
        if type(n_clbits) is not int or n_clbits < 0:
            raise BackendValidationError(
                "pulse engine classical width must be a non-negative int"
            )
        frozen_plan = tuple(plan)
        outcomes = []
        for _ in range(shots):
            context = _ShotContext(
                state=self._runner.initial_state(),
                classical_memory=[0] * n_clbits,
                rng=rng,
            )
            self._execute_shot(frozen_plan, context)
            outcomes.append(self._runner.finish_shot(context))
        return tuple(outcomes)

    def _execute_shot(
        self, plan: tuple[PulsePlanStep, ...], context: _ShotContext
    ) -> None:
        pending: list[PulseBlock] = []

        def flush() -> None:
            if not pending:
                return
            run = place_pulse_run(
                pending,
                boundary_ns=context.time_ns,
                mode=self._placement_mode,
            )
            enabled = tuple(
                _condition_matches(block.condition, context.classical_memory)
                for block in run.blocks
            )
            self._runner.evolve(run, context, enabled)
            context.time_ns = run.end_ns
            pending.clear()

        for step in plan:
            if isinstance(step, PulseBlock):
                if step.condition is not None:
                    flush()
                    pending.append(step)
                    flush()
                else:
                    pending.append(step)
            elif isinstance(step, (MeasurementStep, ResetStep)):
                flush()
                self._runner.execute_boundary(step, context)
            else:
                raise BackendValidationError(
                    "pulse plan contains an unknown execution step"
                )
        flush()
