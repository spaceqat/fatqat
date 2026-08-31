"""Carrier loss for occupancy-aware backends."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import _require_probability


@dataclass(frozen=True)
class Loss:
    """State-independent carrier loss for occupancy-aware backends.

    Loss changes occupancy rather than damping a carrier within the modeled
    Hilbert space. An occupancy-aware backend samples p independently for each
    selected carrier that is present after a matched operation. A hit removes
    that carrier and its correlations. Gates and reset that require an absent
    carrier have no effect; measurement reports erasure, pairing still changes
    connectivity, and Put can load a fresh ground-state carrier.

    Loss can act on any number of operands selected from the matching
    operation. AtomArraySimulator is currently the only backend that supports
    it. Every site on that simulator starts empty and must be loaded with Put,
    independently of whether Loss is attached or whether p is zero. Loss
    attached to Put is sampled after loading, so it models loading failure or
    immediate post-load loss.

    Args:
        p: Per-carrier probability each time a matching operation runs. Must
            be a finite ``int`` or ``float`` other than ``bool`` in ``[0, 1]``.

    Raises:
        ValueError: If p is not a finite real probability in ``[0, 1]``.

    Examples:
        Model loading failure by applying loss after ``Put``:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> noise = fq.NoiseModel()
        >>> noise.add(fq.noise.Loss(p=0.001), operation=ops.Put)

    Attributes:
        p: Per-carrier loss probability each time a matching operation runs.
    """

    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "Loss.p")
