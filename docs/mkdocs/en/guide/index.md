# User guide

Choose the level of detail that answers your question. Every path starts from
the same backend-independent [`Program`][fatqat.Program], so moving from an
algorithm study to a hardware or physics study does not require a second
authoring model.

!!! tip "New to FatQat?"

    [Build, draw, and run a Bell program](quickstart.md), then learn how to
    [write Programs and registers](program.md). When you are ready to compare
    backends, [choose how much physics to model](execution-models.md).

## Choose an execution path

<div class="grid cards fatqat-guide-paths" markdown>

-   ![Five-qubit variational ansatz](../assets/generated/guide/guide-path-algorithm.png){ loading=lazy width="636" height="409" }

    :material-chart-bell-curve-cumulative: **Explore the algorithm**

    Simulate states, measurements, observables, and parameter sweeps, then
    compare ideal and noisy results or tune performance.

    [Start with simulation :material-arrow-right:](simulation.md)

-   ![Hardware topology with supported and unsupported couplings](../assets/generated/guide/guide-path-hardware.png){ loading=lazy width="490" height="400" }

    :material-chip: **Test hardware constraints**

    Check native operations, placement, connectivity, capacity, atom occupancy,
    pairing, and reference noise.

    [Open hardware-profile simulation :material-arrow-right:](hardware-profile-simulation.md)

-   ![Driven-atom spectroscopy heatmap](../assets/generated/guide/guide-path-physics.png){ loading=lazy width="639" height="418" }

    :material-atom: **Follow the physics**

    Follow calibrated transmon gates and direct pulse controls as continuous
    dynamics, with built-in models for transmons and neutral atoms.

    [Open Hamiltonian emulation :material-arrow-right:](hamiltonian-emulation.md)

</div>

[Compare the execution levels](execution-models.md) if you are unsure which
path contains the detail your study needs.

## Continue by task

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **Simulate and analyze**

    ---

    [Simulation](simulation.md) ·
    [Results](interpret-results.md) ·
    [Visualization](visualization.md) ·
    [Ideal and noisy runs](ideal-and-noisy.md) ·
    [Performance](performance.md)

-   :material-memory:{ .lg .middle } **Model hardware and dynamics**

    ---

    [Hardware profiles](hardware-profile-simulation.md) ·
    [Hamiltonian emulation](hamiltonian-emulation.md) ·
    [Transmons](transmon-emulation.md) ·
    [Neutral atoms](neutral-atom-emulation.md)

-   :material-transit-connection-variant:{ .lg .middle } **Connect and diagnose**

    ---

    [OpenQASM and Qiskit](interoperability.md) ·
    [Troubleshooting](troubleshooting.md)

</div>

!!! tip

    The guide teaches complete workflows and the reasoning behind them. Follow a
    link to the [API reference](../api/index.md) when you need a precise signature
    or contract. The [tutorials](../tutorials/index.md) are longer case studies
    built on the same features.
