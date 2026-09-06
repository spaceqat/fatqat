---
template: home.html
hide:
  - navigation
  - toc
description: Build one FatQat Program and choose the physical detail each quantum study needs.
hero:
  eyebrow: Model and execute quantum programs
  title: One Program. Three levels of physical detail.
  summary: >-
    Write a quantum program once.
    Use the same Program to explore algorithm behavior,
    test device constraints, or follow time-dependent physical dynamics.
  primary_action: Build your first Program
  secondary_action: Compare execution levels
  install_action: Install from source
  structure_alt: One FatQat Program runs through general simulation, hardware-profile simulation, or Hamiltonian emulation, with every path returning a Result.
  choice_label: Choose one execution level
  general_title: General simulation
  general_detail: states · counts · noise
  hardware_title: Hardware-profile simulation
  hardware_detail: native gates · connectivity
  hamiltonian_title: Hamiltonian emulation
  hamiltonian_detail: pulses · leakage · dynamics
  visual_note: Start with one Program, then choose the physical detail at run time.
---

> **Development status:** FatQat is under active development, and its interfaces
> may change between releases. Pin an exact version when reproducibility matters.

<!-- Localized content stays in Markdown; the shared Material hero lives in home.html. -->

<div class="grid cards" markdown>

-   :material-code-braces-box:{ .lg .middle } **One quantum program**

    Keep gates, measurements, conditions, parameters, qudits, and direct
    controls in a single `Program`.

-   :material-tune-variant:{ .lg .middle } **Only the detail you need**

    Start with algorithm behavior, add device constraints when needed, or
    follow continuous-time dynamics when the physics matters.

-   :material-transit-connection-variant:{ .lg .middle } **One execution workflow**

    Every backend accepts a `Program` and uses the same `Result`
    interface, while validating what it can realize.

</div>

## One workflow, three execution levels

Every path starts from the same [`Program`][fatqat.Program] and returns familiar
`Job` and `Result` objects. Choose the least physical detail that can answer the
question in front of you.

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General simulation**

    ---

    Inspect states, counts, and channel noise when algorithm behavior is the question.

    [:octicons-arrow-right-24: Study simulation](guide/simulation.md)

-   :material-memory:{ .lg .middle } **Hardware-profile simulation**

    ---

    Add native gates and connectivity when device constraints matter.

    [:octicons-arrow-right-24: Model a hardware profile](guide/hardware-profile-simulation.md)

-   :material-sine-wave:{ .lg .middle } **Hamiltonian emulation**

    ---

    Follow pulse dynamics, leakage, crosstalk, and non-Markovian effects when
    physical behavior matters.

    [:octicons-arrow-right-24: Follow physical dynamics](guide/hamiltonian-emulation.md)

</div>

## One Grover algorithm, three execution levels

This three-qubit Grover search begins with all eight bit strings equally
likely. Two iterations mark `101` and amplify it. The results below show how
three execution levels change the outcome as hardware constraints and physical
dynamics are introduced.

<figure markdown="span">

![A compact three-qubit Grover search uses fused RY and RZ rotations with four Toffoli gates to amplify 101.](assets/generated/home/grover-circuit.png){ loading=lazy width=1100 height=318 }

</figure>

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General `Simulator`**

    ![The ideal Simulator result gives the target outcome 101 a probability of 94.53 percent.](assets/generated/home/grover-general.png){ loading=lazy width=714 height=515 }

    Circuit-level evolution returns `101` with **94.53%** probability.

-   :material-memory:{ .lg .middle } **`SCQubitSimulator`**

    ![The SCQubitSimulator result gives the target outcome 101 a probability of 86.15 percent with 200-microsecond coherence times and additional CZ depolarizing noise of 0.003 on both edges.](assets/generated/home/grover-sc-profile.png){ loading=lazy width=714 height=515 }

    With `T1 = T2 = 200 µs`, we add CZ depolarizing noise with `p = 0.003`. Compiled native-gate simulation then returns
    `101` with **86.15%** probability.

-   :material-sine-wave:{ .lg .middle } **`TransmonEmulator`**

    ![The three-level TransmonEmulator result gives the target outcome 101 a probability of about 68.5 percent with the same coherence times.](assets/generated/home/grover-transmon.png){ loading=lazy width=714 height=515 }

    Calibrated pulses, three physical levels, and the same coherence times
    return `101` with about **68.5%** probability; physical leakage is **0.0446%**.
    Most of the error comes from imperfect `iSWAP` gates.

</div>

??? abstract "Shared algorithm source — `home_grover_program.py`"

    The compact circuit view and fused rotation data live in this visible
    source. The general and Transmon scripts share the rotation Program; the SC
    script builds equivalent QASM and compiles it to the canonical native basis.

    ```python
    --8<-- "docs/mkdocs/figure-sources/home_grover_program.py"
    ```

??? example "Run each execution model independently"

    Each tab is a top-to-bottom script. It uses the algorithm representation
    appropriate to its target, while private plotting details stay out of the
    execution flow.

    === "General `Simulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_general.py"
        ```

    === "`SCQubitSimulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_sc.py"
        ```

    === "`TransmonEmulator`"

        ```python
        --8<-- "docs/mkdocs/figure-sources/home_grover_transmon.py"
        ```

## Encode hardware behavior when needed

Algorithm developers can stay with general simulation. When their work requires
more detail, compiler and hardware developers can add device-specific operations
to a `Program`. The backend validates when those operations are allowed and
tracks their effects.

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] provides one example.
Sites begin empty, so `Put` loads the atoms; `Pair` makes native `CZ` available,
and `Unpair` removes that connection. Noise attached to these operations can
perturb the state or remove an atom from the array.

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
    print(counts)  # {'10': 8}
    ```

[`Loss`][fatqat.noise.Loss] models the second outcome. It removes a present atom
after a matched operation. Occupancy is tracked independently for every shot,
and measuring an empty site returns the erasure digit `2`.

<figure markdown="span">

![An occupied atom undergoes a matched operation, after which Loss either leaves it present with probability one minus p or removes it with probability p; measurement of the empty site returns 2.](assets/generated/guide/atom-loss-lifecycle.svg){ loading=lazy width=960 height=306 }

<figcaption>Loss is sampled after the matched operation and changes per-shot occupancy.</figcaption>

</figure>

??? example "Simulate atom loss"

    `Loss(p=0.1)` is sampled after `RX`. Surviving atoms return `1`; lost
    atoms return `2`.

    ```python
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops

    loss_model = fq.NoiseModel()
    loss_model.add(fq.noise.Loss(p=0.1), operation=ops.RX)

    lossy_atoms = fq.Program(1, 1)
    lossy_atoms.add(ops.Put, 0)
    lossy_atoms.add(ops.RX(np.pi), 0)
    lossy_atoms.measure_all()

    lossy_counts = fq.simulator.AtomArraySimulator(noise=loss_model).run(
        lossy_atoms,
        shots=100,
        simulation_config={"seed": 7},
    ).result().get_counts()
    print(lossy_counts)  # {'1': 86, '2': 14}
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
