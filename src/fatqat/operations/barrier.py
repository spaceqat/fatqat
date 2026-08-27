"""Barrier: compiler-facing no-op frontend operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class BarrierGate(Operation):
    """Mark a compiler boundary without changing the quantum state.

    A barrier has no matrix, measurement result, or noise boundary. It remains
    in the program until lowering, when the built-in simulators discard it. It
    therefore has no effect on simulated states or counts. ``Program.add``
    accepts a condition, but built-in lowering discards that condition with the
    barrier and never evaluates it.

    Add the singleton ``ops.Barrier`` without parentheses. It accepts one or
    more distinct scalar targets; ``RegisterView`` and an empty target tuple
    are rejected by ``Program.add``. This implementation class is not exported
    through ``fatqat.operations.__all__``.

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
