"""Rectangular-grid interaction selection for two-level atom emulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...atom_arrangement import AtomArrangement


@dataclass(frozen=True, init=False)
class GridInteractionPolicy:
    """Select the static rectangular-grid interaction stencil.

    The policy decides which pairs receive the model-defined signed
    ``C6/R^6`` term. It never changes distances or interaction strengths and
    does not make the coefficient time-dependent. Construct policies through
    :meth:`nearest_neighbor` or :meth:`full_pair`; the class has no public
    free-form constructor.

    Attributes:
        mode: ``"nearest_neighbor"`` or ``"full_pair"``.

    Examples:
        >>> from fatqat.emulator import GridInteractionPolicy
        >>> GridInteractionPolicy.nearest_neighbor().mode
        'nearest_neighbor'
        >>> GridInteractionPolicy.full_pair().mode
        'full_pair'
    """

    mode: Literal["nearest_neighbor", "full_pair"]

    @classmethod
    def nearest_neighbor(cls) -> GridInteractionPolicy:
        """Select horizontal and vertical four-neighbor grid edges.

        Edge construction is linear in the number of rectangular sites and
        deliberately omits diagonal and longer-range pairs. This is the
        default policy of :class:`~fatqat.emulator.Atom2LevelEmulator`.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "mode", "nearest_neighbor")
        return instance

    @classmethod
    def full_pair(cls) -> GridInteractionPolicy:
        """Select every unordered pair of arrangement sites.

        This produces ``N(N-1)/2`` interaction terms and is intended as an
        explicit small-system reference mode rather than the large-grid
        default.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "mode", "full_pair")
        return instance


def _nearest_neighbor_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(cols):
            site = row * cols + column
            if column + 1 < cols:
                edges.append((site, site + 1))
            if row + 1 < rows:
                edges.append((site, site + cols))
    return tuple(edges)


def _full_pair_edges(cardinality: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in range(cardinality)
        for second in range(first + 1, cardinality)
    )


def _interaction_edges(
    policy: GridInteractionPolicy, arrangement: AtomArrangement
) -> tuple[tuple[int, int], ...]:
    if not isinstance(policy, GridInteractionPolicy):
        raise TypeError("policy must be a GridInteractionPolicy")
    if not isinstance(arrangement, AtomArrangement):
        raise TypeError("arrangement must be an AtomArrangement")
    if policy.mode == "nearest_neighbor":
        return _nearest_neighbor_edges(arrangement.rows, arrangement.cols)
    return _full_pair_edges(arrangement.cardinality)


__all__ = ["GridInteractionPolicy"]
