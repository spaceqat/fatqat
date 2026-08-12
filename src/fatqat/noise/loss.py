"""Atom-loss descriptor: a classical, Kraus-free channel for the matrix family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Channel
from .catalog import _require_probability


@dataclass(frozen=True)
class AtomLoss(Channel):
    """Ejects an atom from its trap with a flat, state-independent probability.

    Unlike a `Channel` that resolves to Kraus operators, atom loss is classical:
    the atom is either in the trap or not, so it is modelled per shot by a
    partial trace, not by a density-mixing channel. Width-agnostic
    (`_num_subsystems=None`): attached to a multi-qubit gate it rolls one
    independent die per atom.

    Attributes:
        p: Per-atom loss probability for one occurrence, in ``[0, 1]``.
    """

    _num_subsystems: ClassVar[int | None] = None
    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "AtomLoss.p")
