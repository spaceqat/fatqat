"""Non-destructive atom occupancy check into classical slots."""

from __future__ import annotations

from dataclasses import dataclass

from ..registers import RegisterRef


@dataclass(frozen=True)
class OccupancyCheck:
    """Record whether each target atom is present without measuring its state.

    Create this instruction with `fatqat.Program.check_occupancy`. Each target
    is paired with a dimension-2 classical output. A present atom writes ``1``
    and an empty site writes ``0``. Repeated outputs are written in order, so
    the last write wins. Only `fatqat.simulator.AtomArraySimulator` supports
    this instruction.
    """

    targets: tuple[RegisterRef, ...]
    outputs: tuple[RegisterRef, ...]

    def __post_init__(self) -> None:
        if not self.targets or len(self.targets) != len(self.outputs):
            raise ValueError(
                "occupancy check requires equal nonzero numbers of targets and outputs"
            )
        for output in self.outputs:
            if output.register.dim != 2:
                raise ValueError("occupancy check outputs must have dimension 2")
