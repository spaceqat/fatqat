"""Shared normalization for public QuTiP runtime metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _qutip_runtime_details(
    solvers: Iterable[str],
    solver_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Return owned metadata that truthfully summarizes all invoked solvers."""
    names = tuple(sorted(set(solvers)))
    if not names:
        solver: str | tuple[str, ...] = "none"
    elif len(names) == 1:
        solver = names[0]
    else:
        solver = names
    return {
        "solver": solver,
        "solver_options": {} if not names else dict(solver_options),
    }


__all__: list[str] = []
