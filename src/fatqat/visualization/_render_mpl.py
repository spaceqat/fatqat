"""Private Matplotlib renderers for FATQAT visualizations."""

from __future__ import annotations

from typing import Any

from ._viewmodels import _CountsView, _InteractionFrequencyGraph


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


def _render_interaction_frequency(
    graph: _InteractionFrequencyGraph,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a logical-qubit interaction frequency graph."""
    from math import cos, pi, sin

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

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

    count = len(graph.nodes)
    positions: dict[Any, tuple[float, float]] = {}
    if count == 1:
        positions[graph.nodes[0]] = (0.0, 0.0)
    elif count:
        positions = {
            node: (
                cos(2 * pi * index / count),
                sin(2 * pi * index / count),
            )
            for index, node in enumerate(graph.nodes)
        }

    maximum = max((edge.count for edge in graph.edges), default=1)
    node_labels = {node: str(index) for index, node in enumerate(graph.nodes)}
    for edge in graph.edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        axis.add_line(
            Line2D(
                [x1, x2],
                [y1, y2],
                linewidth=1.5 + 5.0 * edge.count / maximum,
                color="C0",
                alpha=0.8,
                zorder=1,
            )
        )
        axis.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            str(edge.count),
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            zorder=3,
        )

    if count:
        xs, ys = zip(*(positions[node] for node in graph.nodes))
        axis.scatter(
            xs,
            ys,
            s=700,
            marker="o",
            color="C1",
            edgecolors="black",
            linewidths=1.2,
            zorder=2,
        )
        for node in graph.nodes:
            x, y = positions[node]
            axis.text(
                x,
                y,
                node_labels[node],
                ha="center",
                va="center",
                zorder=4,
            )

    margin = 1.35 if count > 1 else 1.0
    axis.set_xlim(-margin, margin)
    axis.set_ylim(-margin, margin)
    axis.set_aspect("equal")
    axis.axis("off")
    if title is not None:
        axis.set_title(title)
    if owns_figure:
        figure.tight_layout()
    return figure
