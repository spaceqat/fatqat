"""Carrier-loss declaration for occupancy-aware backends."""

from __future__ import annotations

from dataclasses import dataclass
from .catalog import _require_probability


@dataclass(frozen=True)
class Loss:
    """Remove a present physical carrier with state-independent probability.

    Loss is an occupancy transition, not a retained-space quantum channel. An
    occupancy-aware backend samples it independently for each selected,
    currently present carrier and discards that carrier's correlations on a
    hit. It is width-agnostic over the selected occurrence extent.

    Attributes:
        p: Per-carrier loss probability for one occurrence, in ``[0, 1]``.
    """

    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "Loss.p")
