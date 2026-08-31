"""Private presentation helpers for the homepage Grover comparison."""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from home_grover_program import TARGET, TARGET_INDEX

CIRCUIT_FIGURE = "grover-circuit.png"
GENERAL_FIGURE = "grover-general.png"
OUTCOMES = tuple(f"{value:0{len(TARGET)}b}" for value in range(2 ** len(TARGET)))

INK = "#172033"
MUTED = "#607087"
GRID = "#DDE5EF"
PANEL = "#F7F9FC"
BAR = "#B8C7DB"
TARGET_BAR = "#18AFA4"
TARGET_DARK = "#087A75"

PROGRAM_DRAW_STYLE = {
    "theme": "light",
    "bgcolor": "white",
    "color": INK,
    "wire_color": "#526176",
    "fontsize": 11,
    "gate_margin": 0.11,
    "gate_pad": 0.04,
    "layer_sep": 0.47,
    "wire_sep": 0.48,
}


def style_program_figure(figure, axis):
    """Apply the homepage presentation to an already drawn Program."""
    figure.set_size_inches(13.0, 3.2, forward=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_anchor("W")
    axis.set_title(
        "Grover search", loc="left", fontsize=13.5, fontweight="bold", color=INK
    )
    axis.text(
        1.0,
        1.10,
        "fused 1q rotations  ·  logical Toffoli  ·  2 iterations",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10.5,
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.2",
            "facecolor": PANEL,
            "edgecolor": GRID,
            "linewidth": 0.8,
        },
    )


def draw_distribution(name, probabilities):
    """Draw one result chart on the shared comparison scale."""
    figure, axis = plt.subplots(num=name, figsize=(5.0, 3.6), facecolor="white")
    axis.set_facecolor(PANEL)
    x_positions = np.arange(len(OUTCOMES))
    colors = [
        TARGET_BAR if index == TARGET_INDEX else BAR
        for index in range(len(OUTCOMES))
    ]
    edges = [
        TARGET_DARK if index == TARGET_INDEX else "#9EB0C7"
        for index in range(len(OUTCOMES))
    ]
    bars = axis.bar(
        x_positions,
        probabilities,
        width=0.58,
        color=colors,
        edgecolor=edges,
        linewidth=[1.2 if index == TARGET_INDEX else 0.7 for index in x_positions],
        zorder=3,
    )
    bars[TARGET_INDEX].set_path_effects(
        [
            path_effects.SimplePatchShadow(
                offset=(2, -2), shadow_rgbFace="#0B776F", alpha=0.18
            ),
            path_effects.Normal(),
        ]
    )
    axis.axhline(0.125, color="#7167C7", linewidth=1.1, linestyle=(0, (3, 3)))
    axis.text(
        -0.42,
        0.145,
        "1 / 8",
        color="#6259B4",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.0},
    )
    for index, probability in enumerate(probabilities):
        if index == TARGET_INDEX:
            axis.text(
                index,
                probability + 0.025,
                f"{probability:.2%}",
                color=INK,
                fontsize=11.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=4,
            )
        else:
            axis.text(
                index,
                probability + 0.018,
                f"{probability:.1%}",
                color=MUTED,
                fontsize=8.5,
                ha="center",
                va="bottom",
                zorder=4,
                bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 0.2},
            )
    axis.set_xlim(-0.55, 7.55)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x_positions, OUTCOMES)
    axis.set_yticks((0.0, 0.5, 1.0), ("0%", "50%", "100%"))
    axis.set_ylabel("p(x)", color=MUTED, labelpad=8)
    axis.tick_params(axis="x", colors=INK, labelsize=8.5, length=0, pad=8)
    axis.tick_params(axis="y", colors=MUTED, labelsize=9, length=0, pad=6)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=1)
    axis.xaxis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.8)
    return figure
