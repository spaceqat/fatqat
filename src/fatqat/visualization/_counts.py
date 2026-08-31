"""Prepare Result counts for visualization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._viewmodels import _CountsView


def _prepare_counts(
    counts: Mapping[str, int],
    *,
    stat: str = "counts",
    number_to_keep: int | None = None,
    sort: str = "key",
) -> _CountsView:
    """Validate, select, sort, and scale counts for rendering."""
    if not isinstance(stat, str):
        raise TypeError(f"stat must be a string, got {type(stat).__name__!r}")
    if stat not in {"counts", "frequencies"}:
        raise ValueError(f"stat must be 'counts' or 'frequencies', got {stat!r}")

    if number_to_keep is not None:
        if type(number_to_keep) is not int:
            raise TypeError(
                "number_to_keep must be a positive int or None, "
                f"got {number_to_keep!r}"
            )
        if number_to_keep <= 0:
            raise ValueError(f"number_to_keep must be positive, got {number_to_keep}")

    if not isinstance(sort, str):
        raise TypeError(f"sort must be a string, got {type(sort).__name__!r}")
    if sort not in {"key", "count"}:
        raise ValueError(f"sort must be 'key' or 'count', got {sort!r}")

    if not counts:
        raise ValueError("counts are empty; nothing to draw")

    items: list[tuple[str, int]] = []
    for label, count in counts.items():
        if not isinstance(label, str):
            raise TypeError(
                f"count labels must be strings, got {type(label).__name__!r}"
            )
        if type(count) is not int:
            raise TypeError(f"count values must be ints, got {count!r} for {label!r}")
        if count <= 0:
            raise ValueError(
                f"count values must be positive, got {count} for {label!r}"
            )
        items.append((label, count))

    total = sum(count for _, count in items)
    ranked = sorted(items, key=lambda item: (-item[1], item[0]))
    has_other = number_to_keep is not None and number_to_keep < len(ranked)

    if has_other:
        kept = ranked[:number_to_keep]
        other_count = sum(count for _, count in ranked[number_to_keep:])
    else:
        kept = items
        other_count = 0

    if sort == "key":
        kept = sorted(kept, key=lambda item: item[0])
    else:
        kept = sorted(kept, key=lambda item: (-item[1], item[0]))

    if has_other:
        kept.append(("other", other_count))

    values: tuple[int | float, ...]
    if stat == "frequencies":
        values = tuple(count / total for _, count in kept)
    else:
        values = tuple(count for _, count in kept)

    return _CountsView(
        labels=tuple(label for label, _ in kept),
        values=values,
        total=total,
        stat=stat,
        has_other=has_other,
    )


def _draw_result_counts(
    result: Any,
    *,
    stat: str = "counts",
    number_to_keep: int | None = None,
    sort: str = "key",
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Prepare one Result and dispatch it to the Matplotlib renderer."""
    view = _prepare_counts(
        result.get_counts(),
        stat=stat,
        number_to_keep=number_to_keep,
        sort=sort,
    )
    from ._render_mpl import _render_counts

    return _render_counts(
        view,
        ax=ax,
        title=title,
        figsize=figsize,
    )
