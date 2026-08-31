"""Private Matplotlib renderers for FATQAT visualizations."""

from __future__ import annotations

from typing import Any

from ._viewmodels import _CountsView


def _render_counts(
    view: _CountsView,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render prepared counts as a Matplotlib bar chart."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if ax is not None and figsize is not None:
        raise ValueError("figsize cannot be used together with ax")

    owns_figure = ax is None
    if owns_figure:
        figure = Figure(figsize=figsize)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
    else:
        axis = ax
        figure = axis.get_figure()

    positions = list(range(len(view.labels)))
    axis.bar(positions, view.values, edgecolor="none")
    axis.set_xticks(positions, view.labels)
    axis.set_xlabel("Outcome")
    axis.set_ylabel("Frequency" if view.stat == "frequencies" else "Counts")
    if view.stat == "frequencies":
        axis.set_ylim(0, 1)
    else:
        axis.set_ylim(bottom=0)
    axis.set_axisbelow(True)
    axis.grid(axis="y", linewidth=0.8, alpha=0.7)

    if title is not None:
        axis.set_title(title)

    if len(view.labels) > 8:
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if owns_figure:
        figure.tight_layout()

    return figure
