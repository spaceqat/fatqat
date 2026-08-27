"""Compiler barrier operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class BarrierGate(Operation):
    """Mark a compiler boundary without changing the quantum state.

    A barrier has no matrix, measurement result, or noise boundary. Built-in
    simulators treat it as a no-op, so it has no effect on states or counts.
    :meth:`~fatqat.Program.add` accepts a condition, but built-in simulators
    do not evaluate it.

    Add the singleton ``ops.Barrier`` without parentheses. It accepts one or
    more distinct scalar targets; :class:`~fatqat.RegisterView` and an empty
    target tuple are rejected by :meth:`~fatqat.Program.add`.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Barrier, (0, 1))
    """

    name: ClassVar[str] = "Barrier"
    num_subsystems: ClassVar[int | None] = None


# `Barrier` takes no parameters, so - like the fixed gates and `Reset` - it is
# exported only as a singleton value: `ops.Barrier`, not `ops.Barrier()`.
Barrier = BarrierGate()
