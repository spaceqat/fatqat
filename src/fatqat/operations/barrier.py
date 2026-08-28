"""Compiler barrier operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class BarrierGate(Operation):
    """Mark a compiler boundary without changing the quantum state.

    A barrier has no matrix, measurement result, or noise boundary. Built-in
    simulators ignore it, including any recorded condition, so it has no effect
    on states or counts. Selecting ``ops.Barrier`` in `fatqat.NoiseModel.add`
    raises `ValueError`.

    `fatqat.Program.draw` shows a barrier as a dashed vertical separator across
    its targets rather than as an executable gate box.

    ``Barrier`` accepts one or more distinct scalar targets.
    `fatqat.Program.add` rejects `fatqat.RegisterView` and an empty target
    tuple.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Barrier, (0, 1))
    """

    name: ClassVar[str] = "Barrier"
    num_subsystems: ClassVar[int | None] = None


Barrier = BarrierGate()
