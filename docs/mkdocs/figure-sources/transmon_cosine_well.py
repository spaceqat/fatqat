"""Illustrate the qubit and leakage levels within a Josephson potential well."""

import matplotlib.pyplot as plt
import numpy as np

# The charge-basis Hamiltonian in units of E_J, at zero offset charge.
josephson_to_charging_energy = 50
charges = np.arange(-20, 21)
josephson_coupling = np.full(charges.size - 1, -0.5)
hamiltonian = (
    np.diag(4 * charges**2 / josephson_to_charging_energy)
    + np.diag(josephson_coupling, 1)
    + np.diag(josephson_coupling, -1)
)
levels = np.linalg.eigvalsh(hamiltonian)[:3]

phase = np.linspace(-1.78, 1.78, 481)
fig, ax = plt.subplots(figsize=(6.1, 3.35), num="transmon-cosine-well.svg")
ink = "#314858"
guide = "#a3b1bd"
colors = ("#27699c", "#27699c", "#a94422")
gap_column = 2.18

ax.plot(phase, -np.cos(phase), linewidth=2.2, color=ink)
ax.text(-1.53, 0.23, r"$U(\varphi)=-E_J\cos\varphi$", va="top", fontsize=11, color=ink)
ax.text(2.45, 0.23, r"$\Delta E_{01} > \Delta E_{12}$", ha="center", va="top", fontsize=12, color=ink)

for index, (energy, color) in enumerate(zip(levels, colors)):
    turning_point = np.arccos(-energy)
    ax.hlines(energy, -turning_point, turning_point, linewidth=2.2, color=color)
    ax.hlines(energy, turning_point, gap_column, linewidth=0.8, color=guide, linestyles="dotted")
    ax.text(1.83, energy + 0.026, rf"$|{index}\rangle$", va="bottom", color=color, fontsize=13)

for index, label in enumerate((r"$\Delta E_{01}$", r"$\Delta E_{12}$")):
    lower, upper = levels[index : index + 2]
    color = colors[index + 1]
    ax.annotate(
        "",
        xy=(gap_column, upper),
        xytext=(gap_column, lower),
        arrowprops={"arrowstyle": "<->", "lw": 1.35, "color": color, "shrinkA": 1, "shrinkB": 1},
    )
    ax.text(gap_column + 0.13, (lower + upper) / 2, label, va="center", color=color, fontsize=12)

ax.set(xlabel=r"$\varphi$", ylabel=r"$E/E_J$", xlim=(-1.87, 2.95), ylim=(-1.07, 0.31))
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, -0.5, 0])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(guide)
ax.tick_params(colors=ink, length=3)
fig.tight_layout()
