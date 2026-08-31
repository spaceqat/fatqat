---
template: home.html
hide:
  - navigation
  - toc
description: Build one FatQat Program and choose the physical detail each quantum study needs.
hero:
  eyebrow: Quantum SDK
  title: One Program. Three levels of physical detail.
  summary: >-
    Write a quantum computation once. Then study logical behavior, test device
    constraints, or follow time-dependent physical dynamics without changing
    authoring models.
  primary_action: Build your first Program
  secondary_action: Compare execution levels
  install_action: Install from source
  structure_alt: One FatQat Program runs through general simulation, a hardware profile, or Hamiltonian emulation, with every path returning a Result.
  choice_label: Choose one execution level
  general_title: General simulation
  general_detail: states · counts · noise
  hardware_title: Hardware profile
  hardware_detail: native gates · topology
  hamiltonian_title: Hamiltonian emulation
  hamiltonian_detail: pulses · leakage · dynamics
  visual_note: Author once, then choose the detail at run time.
---

<!-- Localized content stays in Markdown; the shared Material hero lives in home.html. -->

<div class="grid cards" markdown>

-   :material-code-braces-box:{ .lg .middle } **One authoring model**

    Keep gates, measurements, conditions, parameters, qudits, and direct
    controls in a single `Program`.

-   :material-tune-variant:{ .lg .middle } **Only the detail you need**

    Start with algorithm behavior, add device constraints when needed, or
    follow continuous-time dynamics when the physics matters.

-   :material-transit-connection-variant:{ .lg .middle } **One execution workflow**

    Every target accepts a `Program` and uses the same `Job` / `Result`
    interface, while validating what it can realize.

</div>

## One workflow, three execution levels

Every path starts from the same [`Program`][fatqat.Program] and returns familiar
`Job` and `Result` objects. Choose the least physical detail that can answer the
question in front of you.

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General simulation**

    ---

    Inspect states, counts, and noise when logical behavior is the question.

    [:octicons-arrow-right-24: Study simulation](guide/simulation.md)

-   :material-memory:{ .lg .middle } **Hardware-profile simulation**

    ---

    Add native gates and topology when device constraints matter.

    [:octicons-arrow-right-24: Model a hardware profile](guide/hardware-profile-simulation.md)

-   :material-sine-wave:{ .lg .middle } **Hamiltonian emulation**

    ---

    Follow pulses, leakage, and dynamics when physical behavior matters.

    [:octicons-arrow-right-24: Follow physical dynamics](guide/hamiltonian-emulation.md)

</div>

## One Grover Program, three execution models

This three-qubit Grover `Program` begins with all eight bit strings equally
likely. Two iterations mark `101` and amplify it. Its fused `RX` / `RY` / `RZ`
/ `CZ` realization runs unchanged on all three targets below. The two physical
models share the same `T1 = T2 = 200 µs` coherence assumption.

<figure markdown="span">

![A compact three-qubit Grover search uses fused RY and RZ rotations with four logical Toffoli gates to amplify 101.](assets/generated/home/grover-circuit.png){ loading=lazy width=1100 height=318 }

<figcaption>Adjacent single-qubit gates are fused into rotations; Toffoli stays logical. All three backends execute the same fused native Program.</figcaption>

</figure>

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General `Simulator`**

    ![The ideal Simulator result gives the target outcome 101 a probability of 94.53 percent.](assets/generated/home/grover-general.png){ loading=lazy width=714 height=515 }

    Circuit-level evolution returns `101` with **94.53%** probability.

-   :material-memory:{ .lg .middle } **`SCQubitGoogleSimulator`**

    ![The SCQubitGoogleSimulator result gives the target outcome 101 a probability of 86.36 percent with 200-microsecond coherence times and additional CZ depolarizing noise of 0.003 on both edges.](assets/generated/home/grover-google-profile.png){ loading=lazy width=714 height=515 }

    With `T1 = T2 = 200 µs`, we add CZ depolarizing noise with `p = 0.003`
    on both q0–q1 and q1–q2. Native-gate simulation then returns `101` with
    **86.36%** probability.

-   :material-sine-wave:{ .lg .middle } **`TransmonEmulator`**

    ![The three-level TransmonEmulator result gives the target outcome 101 a probability of 68.59 percent with the same coherence times.](assets/generated/home/grover-transmon.png){ loading=lazy width=714 height=515 }

    Calibrated pulses, three physical levels, and the same coherence times
    return `101` with **68.59%** probability; physical leakage is **0.0446%**.

</div>

??? abstract "Shared `Program` — `home_grover_program.py`"

    The compact logical view and the exact fused native Program live in this one
    visible source. Every execution script below imports it unchanged.

    ```python
    --8<-- "docs/mkdocs/figure-sources/home_grover_program.py"
    ```

??? example "Run each execution model independently"

    Each tab is a top-to-bottom script. It imports the shared Program above,
    while private plotting details stay out of the execution flow.

    === "General `Simulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_general.py"
        ```

    === "`SCQubitGoogleSimulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_google.py"
        ```

    === "`TransmonEmulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_transmon.py"
        ```

## Program a reconfigurable atom array

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] brings occupancy and
changing connectivity into the same `Program` model. Every declared site
starts empty: `Put` establishes which sites are occupied, `Pair` makes native
`CZ` legal, and `Unpair` removes that eligibility. Optional movement noise can
attach to those operations explicitly.

<figure markdown="span">

![Two occupied atoms begin separated, pair so that CZ becomes legal, and unpair while optional depolarizing noise follows the pairing operations.](assets/generated/guide/atom-pairing-lifecycle.svg){ loading=lazy width=960 height=306 }

<figcaption>Occupancy stays explicit while pairing changes which native interactions are legal.</figcaption>

</figure>

??? example "Run the atom-array Program"

    ```python
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops

    atoms = fq.Program(2, 2)
    atoms.add(ops.Put, (0, 1))
    atoms.add(ops.Pair, (0, 1))
    atoms.add(ops.RX(np.pi), 0)
    atoms.add(ops.CZ, (0, 1))
    atoms.add(ops.Unpair, (0, 1))
    atoms.measure_all()

    counts = fq.simulator.AtomArraySimulator().run(
        atoms,
        shots=8,
        simulation_config={"seed": 7},
    ).result().get_counts()
    print(counts)  # {'01': 8}
    ```

[:octicons-arrow-right-24: Track occupancy, pairing, and loss](guide/hardware-profile-simulation.md#atom-occupancy-and-pairing)

## Explore the documentation

<div class="grid cards" markdown>

-   :material-play-circle-outline:{ .lg .middle } **Quickstart**

    ---

    Build, draw, and run a first Program.

    [:octicons-arrow-right-24: Start building](guide/quickstart.md)

-   :material-book-open-page-variant-outline:{ .lg .middle } **User guide**

    ---

    Learn concepts and complete workflows.

    [:octicons-arrow-right-24: Read the guide](guide/index.md)

-   :material-flask-outline:{ .lg .middle } **Tutorials**

    ---

    Explore executable algorithm and physics studies.

    [:octicons-arrow-right-24: Run a tutorial](tutorials/index.md)

-   :material-format-list-bulleted:{ .lg .middle } **API reference**

    ---

    Find exact signatures and validation contracts.

    [:octicons-arrow-right-24: Look up an API](api/index.md)

</div>
