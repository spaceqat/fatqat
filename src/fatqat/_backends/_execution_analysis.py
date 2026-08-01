"""Private, backend-neutral analysis for terminal-measurement fast paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .steps import MeasurementStep


@dataclass(frozen=True)
class _OperationExecutionFacts:
    """Execution facts needed to classify one non-measurement plan step.

    ``target_indices=None`` means the step has no engine-subsystem ownership
    metadata.  It is safe before every measurement, but must conservatively
    force per-shot replay after one.
    """

    target_indices: tuple[int, ...] | None
    is_conditioned: bool
    forces_per_shot: bool = False


def _analyze_terminal_measurements(
    plan: Iterable[Any],
    operation_facts: Callable[[Any], _OperationExecutionFacts],
) -> tuple[bool, tuple[MeasurementStep, ...]]:
    """Classify whether one evolution can precede terminal measurement sampling.

    A condition, an intrinsically stochastic operation, or an operation that
    touches a previously measured subsystem requires per-shot replay.  Later
    deterministic operations on disjoint subsystems commute with deferred
    measurement and may remain on the fast path.
    """
    measured_indices: set[int] = set()
    measurements: list[MeasurementStep] = []
    for step in plan:
        if isinstance(step, MeasurementStep):
            measured_indices.update(step.measured_indices)
            measurements.append(step)
            continue
        facts = operation_facts(step)
        if facts.forces_per_shot or facts.is_conditioned:
            return True, ()
        if measured_indices and (
            facts.target_indices is None
            or measured_indices.intersection(facts.target_indices)
        ):
            return True, ()
    return False, tuple(measurements)
