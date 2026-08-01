"""Shared private scalar expansion for program instructions with register views."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import BackendValidationError
from ..operations import Measurement
from ..program import AppliedOperation
from ..registers import RegisterView, _view_members

ProgramInstruction = AppliedOperation | Measurement


def _expand_grouped_operation(
    step: AppliedOperation,
) -> tuple[AppliedOperation, ...]:
    """Expand one view-bearing operation into scalar applied operations."""
    target_members = tuple(
        _view_members(target) if isinstance(target, RegisterView) else (target,)
        for target in step.targets
    )
    if not any(isinstance(target, RegisterView) for target in step.targets):
        return (step,)

    name = type(step.operation).__name__
    if len(target_members) == 1:
        emissions = [(member,) for member in target_members[0]]
    elif len(target_members) == 2:
        first, second = target_members
        emissions = list(zip(first, second))
    else:
        raise BackendValidationError(
            f"{name} cannot expand a view target at arity {len(target_members)}"
        )

    return tuple(
        AppliedOperation(
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
        if isinstance(step, AppliedOperation) and any(
            isinstance(target, RegisterView) for target in step.targets
        ):
            broken.extend(_expand_grouped_operation(step))
        else:
            broken.append(step)
    return tuple(broken)
