"""Private data-only view models used by visualization renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _CountsView:
    """Prepared counts data independent of Matplotlib."""

    labels: tuple[str, ...]
    values: tuple[int | float, ...]
    total: int
    stat: str
    has_other: bool = False
