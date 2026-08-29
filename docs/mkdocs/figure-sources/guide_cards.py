"""Canonical source for the three visual guide-path cards."""

from __future__ import annotations

from pathlib import Path


def render(output: Path) -> tuple[str, ...]:
    """Render the guide-path cards and return their output filenames."""

    output.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.7, 2.4))
    for qubit in range(5):
        ax.plot((0.25, 7.75), (qubit, qubit), color="0.55", linewidth=1.0)
        ax.text(0.05, qubit, f"q{qubit}", ha="right", va="center", fontsize=8)
    for column, label in ((1.0, "RX"), (2.0, "RY"), (5.9, "RX"), (6.9, "RY")):
        for qubit in range(5):
            ax.text(
                column,
                qubit,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                bbox={"boxstyle": "round,pad=0.28", "fc": "C0", "ec": "C0"},
            )
    for column, pairs in ((3.0, ((0, 1), (2, 3))), (4.1, ((1, 2), (3, 4)))):
        for lower, upper in pairs:
            ax.plot((column, column), (lower, upper), color="C3", linewidth=1.4)
            ax.scatter((column, column), (lower, upper), s=24, color="C3", zorder=3)
    ax.axvline(5.0, color="0.45", linestyle=(0, (2, 2)), linewidth=1.0)
    ax.text(5.0, 4.45, "layer", ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_title("5-qubit VQA ansatz", fontsize=10.5, pad=4)
    ax.set(xlim=(-0.25, 8.0), ylim=(4.65, -0.65))
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    name = "guide-path-algorithm.png"
    fig.savefig(
        output / name,
        dpi=144,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "fatqat guide-card renderer"},
    )
    plt.close(fig)
    rendered.append(name)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    import fatqat as fq
    import fatqat.operations as ops

    profile = fq.simulator.SCQubitGoogleSimulator(
        grid_size=(3, 4),
        runtime="numpy",
    )
    gate_map = profile.implementation_map
    edges = {
        tuple(sorted(edge))
        for edge in gate_map.device_operands_for(ops.CZ)
    }
    native_pair = (9, 10)
    diagonal_pair = (1, 6)
    assert gate_map.supports(ops.CZ, device_operands=native_pair)
    assert not gate_map.supports(ops.CZ, device_operands=diagonal_pair)

    positions = {site: (site % 4, 2 - site // 4) for site in range(12)}
    fig, ax = plt.subplots(figsize=(4.2, 2.55))

    chip = FancyBboxPatch(
        (-0.38, -0.38),
        3.76,
        2.76,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="C0",
        edgecolor="0.72",
        linewidth=1.0,
        alpha=0.07,
        zorder=0,
    )
    ax.add_patch(chip)

    for left, right in edges:
        x = (positions[left][0], positions[right][0])
        y = (positions[left][1], positions[right][1])
        ax.plot(x, y, color="0.76", linewidth=1.6, zorder=1)

    for site, (x, y) in positions.items():
        ax.scatter(x, y, s=500, color="C0", alpha=0.10, edgecolor="none", zorder=2)
        ax.scatter(
            x,
            y,
            s=360,
            facecolor="white",
            edgecolor="0.42",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(x, y, str(site), ha="center", va="center", fontsize=8.5, zorder=4)

    native_x = tuple(positions[site][0] for site in native_pair)
    native_y = tuple(positions[site][1] for site in native_pair)
    ax.plot(native_x, native_y, color="C2", linewidth=4.0, zorder=2)
    ax.scatter(
        native_x,
        native_y,
        s=610,
        facecolor="none",
        edgecolor="C2",
        linewidth=2.2,
        zorder=5,
    )
    ax.text(
        1.5,
        -0.30,
        "native CZ",
        color="C2",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    rejected_x = tuple(positions[site][0] for site in diagonal_pair)
    rejected_y = tuple(positions[site][1] for site in diagonal_pair)
    ax.plot(
        rejected_x,
        rejected_y,
        color="C3",
        linestyle=(0, (3, 2)),
        linewidth=2.4,
        zorder=4,
    )
    ax.scatter(
        rejected_x,
        rejected_y,
        s=610,
        facecolor="none",
        edgecolor="C3",
        linewidth=2.2,
        zorder=5,
    )
    ax.text(
        1.5,
        1.5,
        "X",
        color="C3",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        zorder=6,
    )
    ax.text(1.78, 1.62, "not connected", color="C3", fontsize=9.5)

    ax.set_title("CZ placement on a 3 x 4 device", fontsize=11, pad=4)
    ax.set(xlim=(-0.48, 3.48), ylim=(-0.48, 2.48), aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    name = "guide-path-hardware.png"
    fig.savefig(
        output / name,
        dpi=144,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "fatqat guide-card renderer"},
    )
    plt.close(fig)
    rendered.append(name)

    import matplotlib.pyplot as plt
    import numpy as np

    import fatqat as fq
    import fatqat.operations as ops

    model = fq.emulator.Atom2LevelModel.from_document(
        fq.emulator.load_model_document("atom2level.reference")
    )
    arrangement = fq.emulator.AtomArrangement.rectangular(
        1,
        1,
        spacing=6.0,
    )
    backend = fq.emulator.Atom2LevelEmulator(
        model,
        arrangement=arrangement,
    )
    omega = 2.0 * np.pi
    durations = np.linspace(0.0, 1.5, 31)
    detunings = np.linspace(-3.0 * omega, 3.0 * omega, 25)
    step = durations[1] - durations[0]
    population = np.zeros((len(detunings), len(durations)))
    ground_state = np.array([1.0, 0.0], dtype=complex)

    for row, detuning in enumerate(detunings):
        drive = fq.emulator.PulseControl(
            model.control.drive(),
            fq.emulator.SampledWaveform((0.0, step), (omega, omega)),
        )
        offset = fq.emulator.PulseControl(
            model.control.detuning(),
            fq.emulator.SampledWaveform((0.0, step), (detuning, detuning)),
        )
        program = fq.Program(arrangement.num_sites)
        program.add(ops.PulseOperation(step, (drive, offset)))
        propagator = backend.propagator(program)

        state = ground_state.copy()
        for column in range(1, len(durations)):
            state = propagator @ state
            population[row, column] = abs(state[1]) ** 2

    resonance = population[len(detunings) // 2]
    np.testing.assert_allclose(
        resonance,
        np.sin(omega * durations / 2.0) ** 2,
        atol=1e-7,
    )
    np.testing.assert_allclose(population, population[::-1], atol=1e-7)

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    image = ax.pcolormesh(
        durations,
        detunings / (2.0 * np.pi),
        population,
        shading="gouraud",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.contour(
        durations,
        detunings / (2.0 * np.pi),
        population,
        levels=(0.25, 0.5, 0.75),
        colors="white",
        linewidths=0.45,
        alpha=0.42,
    )
    ax.axhline(0.0, color="white", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(
        1.47,
        0.18,
        "resonance",
        color="white",
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    ax.set(
        xlabel=r"pulse duration ($\mu$s)",
        ylabel=r"detuning $\Delta / 2\pi$ (MHz)",
        xlim=(0.0, 1.5),
        ylim=(-3.0, 3.0),
        xticks=(0.0, 0.5, 1.0, 1.5),
        yticks=(-3.0, 0.0, 3.0),
    )
    ax.set_title("Driven-atom spectroscopy", fontsize=11, pad=4)
    ax.tick_params(labelsize=9.5)
    ax.xaxis.label.set_size(10.5)
    ax.yaxis.label.set_size(10.5)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, aspect=18)
    colorbar.set_label(r"$P_r$", rotation=0, labelpad=8, fontsize=10.5)
    colorbar.set_ticks((0.0, 0.5, 1.0))
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.35)
    name = "guide-path-physics.png"
    fig.savefig(
        output / name,
        dpi=144,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "fatqat guide-card renderer"},
    )
    plt.close(fig)
    rendered.append(name)

    return tuple(rendered)
