"""Draw the per-shot atom-loss lifecycle used across the docs."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


fig, ax = plt.subplots(figsize=(7.6, 2.35), num="atom-loss-lifecycle.svg")


def draw_atom(x, y):
    """Draw one occupied atom site."""

    ax.scatter(
        (x,),
        (y,),
        s=610,
        facecolor="C0",
        edgecolor="white",
        linewidth=2.2,
        zorder=3,
    )
    ax.text(
        x,
        y,
        "atom",
        color="white",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        zorder=4,
    )


def draw_empty_site(x, y):
    """Draw one empty trap site."""

    ax.scatter(
        (x,),
        (y,),
        s=610,
        facecolor="white",
        edgecolor="0.55",
        linewidth=1.8,
        linestyle="--",
        zorder=3,
    )


def arrow(start, end, *, label, color="0.40", label_offset=0.13):
    """Draw a labeled transition."""

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.6},
    )
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    ax.text(
        midpoint[0],
        midpoint[1] + label_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
    )


draw_atom(0.75, 0.0)
draw_atom(2.55, 0.0)

arrow((1.08, 0.0), (2.22, 0.0), label="matched operation")
arrow((2.88, 0.0), (3.72, 0.0), label="then")

loss_box = FancyBboxPatch(
    (3.72, -0.28),
    1.05,
    0.56,
    boxstyle="round,pad=0.08",
    facecolor="C1",
    edgecolor="white",
    linewidth=1.4,
    zorder=3,
)
ax.add_patch(loss_box)
ax.text(
    4.245,
    0.0,
    "Loss(p)",
    color="white",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    zorder=4,
)

draw_atom(6.35, 0.48)
draw_empty_site(6.35, -0.48)

arrow((4.85, 0.12), (6.02, 0.43), label="1 − p", color="C2", label_offset=0.10)
arrow((4.85, -0.12), (6.02, -0.43), label="p", color="C3", label_offset=-0.24)

ax.text(0.75, -0.62, "occupied", ha="center", va="top", fontsize=9.5)
ax.text(2.55, -0.62, "operation applied", ha="center", va="top", fontsize=9.5)
ax.text(6.78, 0.48, "present", ha="left", va="center", fontsize=9.5)
ax.text(6.78, -0.48, "empty · measure 2", ha="left", va="center", fontsize=9.5)

ax.set_title("Loss changes occupancy after a matched operation", fontsize=12, pad=4)
ax.set(xlim=(0.0, 7.6), ylim=(-0.96, 1.05))
ax.axis("off")
fig.tight_layout(pad=0.3)