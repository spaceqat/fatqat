---
template: home.html
hide:
  - navigation
  - toc
description: Explore quantum algorithms, hardware constraints, and physical dynamics with FatQat, a Python package for quantum simulation and emulation.
hero:
  eyebrow: A Python toolkit for quantum simulation and emulation
  title: One Program. Three levels of physical detail.
  summary: >-
    Explore algorithms, check hardware constraints, and study pulse-driven dynamics.
    FatQat brings these tasks together with a shared Program representation
    and consistent interfaces for running computations and reading results.
  primary_action: Build your first Program
  secondary_action: Choose a simulation approach
  install_action: Install from source
  structure_alt: One FatQat Program runs through general simulation, hardware-profile simulation, or Hamiltonian emulation, with every path returning a Result.
  choice_label: Choose the physical detail you need
  general_title: General simulation
  general_detail: states · counts · noise
  hardware_title: Hardware-profile simulation
  hardware_detail: native gates · connectivity
  hamiltonian_title: Hamiltonian emulation
  hamiltonian_detail: pulses · leakage · dynamics
  visual_note: A shared Program interface, with operations and controls suited to each backend.
---

!!! warning "Under active development"

    FatQat is under active development, and its interfaces may change between
    releases. **Pin an exact version when reproducibility matters.**

<!-- Localized content stays in Markdown; the shared Material hero lives in home.html. -->

<div class="grid cards" markdown>

-   :material-code-braces-box:{ .lg .middle } **More than a gate circuit**

    Express gates, measurements, classical conditions, and parameters in a
    `Program`, alongside qudits and direct physical controls.

-   :material-tune-variant:{ .lg .middle } **Only the detail you need**

    Study an algorithm at the circuit level, check whether operations obey
    device rules, or follow the dynamics during a control pulse.

-   :material-transit-connection-variant:{ .lg .middle } **One execution workflow**

    Run computations and inspect their results through consistent `Job` and
    `Result` interfaces. Each backend checks that the `Program` uses operations
    it supports.

</div>

## Three ways to explore a quantum computation

Choose a model around the question you want to answer. All three approaches
use [`Program`][fatqat.Program], but the supported operations and required
controls depend on the backend. A gate circuit may need compilation or pulse
definitions before it can run on a hardware model.

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General simulation**

    ---

    Understand an algorithm through quantum states and measurement outcomes,
    and explore how noise changes its behavior.

    [:octicons-arrow-right-24: Study simulation](guide/simulation.md)

-   :material-memory:{ .lg .middle } **Hardware-profile simulation**

    ---

    Check supported gates and connectivity, or track atom loading, pairing,
    and loss as part of the computation.

    [:octicons-arrow-right-24: Model a hardware profile](guide/hardware-constraints.md)

-   :material-sine-wave:{ .lg .middle } **Hamiltonian emulation**

    ---

    Follow pulse dynamics, leakage, crosstalk, and non-Markovian effects when
    physical behavior matters.

    [:octicons-arrow-right-24: Follow physical dynamics](guide/hamiltonian-emulation.md)

</div>

## One Grover algorithm, three models

How does an algorithm behave when it meets a device model? This three-qubit
Grover search starts with all eight bit strings equally likely, then marks and
amplifies `101` over two iterations. Starting from the same algorithm, the
examples below compare ideal circuit evolution, noisy native gates, and
calibrated control pulses. Each model reveals different effects on the result.

<figure markdown="span">

![A compact three-qubit Grover search uses fused RY and RZ rotations with four Toffoli gates to amplify 101.](assets/generated/home/grover-circuit.png){ loading=lazy width=1100 height=318 }

</figure>

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **General `Simulator`**

    ![The ideal Simulator result gives the target outcome 101 a probability of 94.53 percent.](assets/generated/home/grover-general.png){ loading=lazy width=714 height=515 }

    Circuit-level evolution returns `101` with **94.53%** probability.

-   :material-memory:{ .lg .middle } **`SCQubitSimulator`**

    ![The SCQubitSimulator result gives the target outcome 101 a probability of 81.65 percent with 200-microsecond coherence times and additional CZ depolarizing noise of 0.003 on both edges.](assets/generated/home/grover-sc-profile.png){ loading=lazy width=714 height=515 }

    With `T1 = T2 = 200 µs`, we add CZ depolarizing noise with `p = 0.003`. Compiled native-gate simulation then returns
    `101` with **81.65%** probability.

-   :material-sine-wave:{ .lg .middle } **`TransmonEmulator`**

    ![The three-level TransmonEmulator result gives the target outcome 101 a probability of about 74.02 percent with the same coherence times.](assets/generated/home/grover-transmon.png){ loading=lazy width=714 height=515 }

    Calibrated pulses, three physical levels, and the same coherence times
    return `101` with about **74.02%** probability; physical leakage is **0.0669%**.

</div>

With the fixed routing seed, compilation produces **51 CZ**, **172 SX**, and
**383 RZ** gates. These results use the same compiled circuit for both hardware
models; the SC noise channels and the Transmon pulse dynamics model different
physical effects. The Transmon model also uses 60 ns CZ pulses, while the SC
coherence channels use a 50 ns CZ duration. Compilation preserves ideal
algorithm behavior, but the gate decomposition and routing affect noisy results.

??? abstract "Shared algorithm source — `home_grover_program.py`"

    All three scripts start from the same compact `LogicalProgram` shown above.
    General simulation runs it directly; the SC and Transmon scripts use
    `compile_to_sc()` to decompose its Toffoli gates and route the circuit onto
    the three-site line. The routing seed is fixed for reproducible results.
    The shared axis helper restores logical qubit order when reading a
    compiled final state.

    ```python
    --8<-- "docs/mkdocs/figure-sources/home_grover_program.py"
    ```

??? example "Run each execution model independently"

    Each tab is a top-to-bottom script. The general script also verifies that
    ideal compiled execution matches the source state up to a global phase.
    The Transmon script binds compiled sites to model subsystems and supplies
    X/SX pulse definitions using its calibrated RX recipe.

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

## Make hardware operations part of the program

Some hardware behavior changes more than the quantum state: atoms can be
loaded or lost, and the available interactions can change during a computation.
A `Program` can express these device-specific operations alongside quantum
gates. The backend tracks their effects and checks which operations are
allowed at each step.

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
        execution_config={"seed": 7},
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
        execution_config={"seed": 7},
    ).result().get_counts()
    print(lossy_counts)  # {'1': 86, '2': 14}
    ```
[:octicons-arrow-right-24: Track occupancy, pairing, and loss](guide/atom-array-simulation.md#atom-occupancy-and-pairing)

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

-   :material-flask-outline:{ .lg .middle } **Case studies**

    ---

    Explore executable algorithm and physics studies.

    [:octicons-arrow-right-24: Explore case studies](tutorials/index.md)

-   :material-format-list-bulleted:{ .lg .middle } **API reference**

    ---

    Find exact signatures and validation contracts.

    [:octicons-arrow-right-24: Look up an API](api/index.md)

</div>
