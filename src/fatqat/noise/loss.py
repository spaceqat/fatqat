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
    hit. Absence persists until an explicit refill operation. It is
    width-agnostic over the selected occurrence extent and is supported only
    by backends with carrier-occupancy semantics.

    Args:
        p: Per-carrier loss probability for one matched occurrence, in
            ``[0, 1]``.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]`` or is not a finite real
            probability.

    Examples:
        Apply independent carrier loss after each matched ``RX`` occurrence:

        >>> import fatqat as fq
        >>> noise = fq.NoiseModel()
        >>> noise.add(fq.noise.Loss(p=0.001), operation=fq.ops.RX)

    Attributes:
        p: Per-carrier loss probability for one occurrence.
    """

    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "Loss.p")
