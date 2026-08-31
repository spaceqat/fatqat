"""Draw the atom-array occupancy and pairing lifecycle used across the docs."""

import matplotlib.pyplot as plt


stage_centers = (0.8, 2.8, 4.8, 6.8)
separations = (1.00, 0.42, 0.42, 1.00)
stage_labels = ("occupied", "paired", "CZ eligible", "unpaired")

fig, ax = plt.subplots(figsize=(7.6, 2.35), num="atom-pairing-lifecycle.svg")


def draw_atoms(center, separation, *, connected=False, gate=False, noisy=False):
    """Draw one stage of the two-atom lifecycle."""

    positions = (center - separation / 2.0, center + separation / 2.0)
    if noisy:
        ax.scatter(
            positions,
            (0.0, 0.0),
            s=1050,
            color="C1",
            alpha=0.13,
            edgecolor="none",
            zorder=1,
        )
    if connected:
        ax.plot(
            positions,
            (0.0, 0.0),
            color="C2" if not gate else "C4",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=2,
        )
    ax.scatter(
        positions,
        (0.0, 0.0),
        s=610,
        facecolor="C0",
        edgecolor="white",
        linewidth=2.2,
        zorder=3,
    )
    for atom, position in enumerate(positions):
        ax.text(
            position,
            0.0,
            str(atom),
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )
    if gate:
        ax.text(
            center,
            0.0,
            "CZ",
            color="white",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "C4",
                "edgecolor": "white",
                "linewidth": 1.2,
            },
            zorder=5,
        )


draw_atoms(stage_centers[0], separations[0])
draw_atoms(stage_centers[1], separations[1], connected=True, noisy=True)
draw_atoms(stage_centers[2], separations[2], connected=True, gate=True)
draw_atoms(stage_centers[3], separations[3], noisy=True)

transitions = (
    (stage_centers[0], stage_centers[1], "Pair", True),
    (stage_centers[1], stage_centers[2], "CZ", False),
    (stage_centers[2], stage_centers[3], "Unpair", True),
)
for start, end, label, noisy in transitions:
    ax.annotate(
        "",
        xy=(end - 0.62, 0.72),
        xytext=(start + 0.62, 0.72),
        arrowprops={"arrowstyle": "->", "color": "0.40", "linewidth": 1.6},
    )
    midpoint = (start + end) / 2.0
    ax.text(
        midpoint,
        0.88,
        label,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="C0" if label != "CZ" else "C4",
    )
    if noisy:
        ax.text(
            midpoint,
            0.57,
            "+ depolarizing",
            ha="center",
            va="top",
            fontsize=8.5,
            color="C1",
        )

for center, label in zip(stage_centers, stage_labels):
    ax.text(center, -0.66, label, ha="center", va="top", fontsize=9.5)

ax.set_title("Occupancy stays present while pairing changes", fontsize=12, pad=4)
ax.set(xlim=(0.0, 7.6), ylim=(-0.96, 1.24))
ax.axis("off")
fig.tight_layout(pad=0.3)
