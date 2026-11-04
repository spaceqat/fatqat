"""Draw the four two-photon histories at a beam splitter."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

fig, ax = plt.subplots(figsize=(9.0, 3.6), num="hong-ou-mandel-histories.svg")

ARM = 0.9
SPLIT = 0.07  # separates two photons that share an output
PHOTON_A = "#2f6fbf"
PHOTON_B = "#e07b39"
CANCEL = "#c0392b"
TOGETHER = "#2e8b57"


def group(x0, x1, color, title):
    """Shade a group of histories and title it."""

    ax.add_patch(
        FancyBboxPatch(
            (x0, -2.05),
            x1 - x0,
            3.55,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor=color,
            edgecolor="none",
            alpha=0.08,
            zorder=0,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        1.28,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=color,
    )


def splitter(cx, cy):
    """Draw a glass beam splitter along the diagonal."""

    half, width = 0.42, 0.06
    ax.add_patch(
        Polygon(
            [
                (cx - half - width, cy - half + width),
                (cx + half - width, cy + half + width),
                (cx + half + width, cy + half - width),
                (cx - half + width, cy - half - width),
            ],
            closed=True,
            facecolor="#cfe3f3",
            edgecolor="#6d8fa8",
            linewidth=1.0,
            zorder=4,
        )
    )


def photon_path(start, corner, end, color):
    """Draw a photon entering at ``start`` and leaving towards ``end``."""

    ax.plot(
        (start[0], corner[0], end[0]),
        (start[1], corner[1], end[1]),
        color=color,
        linewidth=2.4,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    ax.annotate(
        "",
        xy=end,
        xytext=(
            end[0] - 0.6 * (end[0] - corner[0]),
            end[1] - 0.6 * (end[1] - corner[1]),
        ),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 0,
            "mutation_scale": 15,
        },
        zorder=3,
    )
    ax.add_patch(
        Circle(start, 0.1, facecolor=color, edgecolor="white", linewidth=1.2, zorder=5)
    )


def history(cx, *, first_out, second_out, outcome, caption, color):
    """Draw one history: photon A enters from the left, photon B from below.

    An output is ``"right"`` (mode 0) or ``"top"`` (mode 1).
    """

    cy = 0.0
    shared = first_out == second_out
    for color_, start, out, shift in (
        (PHOTON_A, (cx - ARM, cy), first_out, SPLIT if shared else 0.0),
        (PHOTON_B, (cx, cy - ARM), second_out, -SPLIT if shared else 0.0),
    ):
        if out == "right":
            corner, end = (cx, cy + shift), (cx + ARM, cy + shift)
        else:
            corner, end = (cx + shift, cy), (cx + shift, cy + ARM)
        if start[1] == cy and out == "right":
            start = (start[0], cy + shift)
        elif start[0] == cx and out == "top":
            start = (cx + shift, start[1])
        photon_path(start, corner, end, color_)
    splitter(cx, cy)
    ax.text(
        cx,
        -1.3,
        outcome,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.2,
        },
    )
    ax.text(cx, -1.72, caption, ha="center", va="center", fontsize=9, color="0.30")


group(-1.35, 4.15, CANCEL, r"Same outcome: $\cos 2\theta$, zero at 50:50")
group(4.65, 10.15, TOGETHER, "Photons leave together")

history(
    0.25,
    first_out="right",
    second_out="top",
    outcome='"11"',
    caption=r"both pass: $\cos\theta\cdot\cos\theta$",
    color=CANCEL,
)
history(
    2.95,
    first_out="top",
    second_out="right",
    outcome='"11"',
    caption=r"both reflect: $\sin\theta\cdot(-\sin\theta)$",
    color=CANCEL,
)
history(
    6.05,
    first_out="right",
    second_out="right",
    outcome='"20"',
    caption="both in mode 0",
    color=TOGETHER,
)
history(
    8.75,
    first_out="top",
    second_out="top",
    outcome='"02"',
    caption="both in mode 1",
    color=TOGETHER,
)
ax.text(1.6, 0.0, "+", ha="center", va="center", fontsize=22, color=CANCEL)

ax.text(
    0.25 - ARM - 0.14,
    0.0,
    "mode 0",
    ha="right",
    va="center",
    fontsize=8.5,
    color=PHOTON_A,
)
ax.text(
    0.25 + 0.14,
    -ARM,
    "mode 1",
    ha="left",
    va="center",
    fontsize=8.5,
    color=PHOTON_B,
)

ax.set(xlim=(-1.45, 10.25), ylim=(-2.15, 1.6))
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)
