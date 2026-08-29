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
  structure_alt: One FatQat Program runs through general simulation, a hardware profile, or Hamiltonian emulation, with every path returning a Job and Result.
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

## See a complete workflow

<div class="grid" markdown>

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()
```

<figure markdown="span">

![A bar chart containing only the correlated 00 and 11 Bell-state outcomes.](assets/generated/guide/quickstart-counts.png){ loading=lazy width="763" height="464" }

<figcaption>One thousand seeded shots return only correlated outcomes.</figcaption>

</figure>

</div>

This Bell Program is a complete circuit-level workflow. The same authoring
object can also carry reusable parameters, mixed local dimensions, classical
conditions, and direct physical controls.

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
