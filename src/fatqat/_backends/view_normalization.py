"""Shared private scalar expansion for program instructions with register views."""

from __future__ import annotations

from collections.abc import Iterable

from ..operations import Measurement
from ..program import _AppliedOperation
from ..registers import RegisterView, _view_members

ProgramInstruction = _AppliedOperation | Measurement


def _expand_grouped_operation(
    step: _AppliedOperation,
) -> tuple[_AppliedOperation, ...]:
    """Expand one view-bearing operation into scalar applied operations."""
    target_members = tuple(
        _view_members(target) if isinstance(target, RegisterView) else (target,)
        for target in step.targets
    )
    if not any(isinstance(target, RegisterView) for target in step.targets):
        return (step,)

    emissions = zip(*target_members, strict=True)

    return tuple(
        _AppliedOperation(
            operation=step.operation,
            targets=tuple(targets),
            condition=step.condition,
        )
        for targets in emissions
    )


def _break_grouped_operations(
    operations: Iterable[ProgramInstruction],
) -> tuple[ProgramInstruction, ...]:
    """Return a scalar-only instruction stream without mutating the program."""
    broken: list[ProgramInstruction] = []
    for step in operations:
        if isinstance(step, _AppliedOperation) and any(
            isinstance(target, RegisterView) for target in step.targets
        ):
            broken.extend(_expand_grouped_operation(step))
        else:
            broken.append(step)
    return tuple(broken)
