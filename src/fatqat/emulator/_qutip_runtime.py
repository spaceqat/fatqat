"""Shared normalization for public QuTiP runtime metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _qutip_runtime_details(
    solvers: Iterable[str],
    solver_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return invoked solvers and the options FatQAT overrides."""
    names = tuple(sorted(set(solvers)))
    if not names:
        solver: str | tuple[str, ...] = "none"
    elif len(names) == 1:
        solver = names[0]
    else:
        solver = names
    return {
        "solver": solver,
        "solver_options": {} if not names else dict(solver_overrides),
    }


__all__: list[str] = []
