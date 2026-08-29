---
hide:
  - navigation
  - toc
---

<!-- Material-specific landing page; keep the product story aligned with docs/sphinx/index.md. -->

<div class="fatqat-home" markdown>

<section class="fatqat-home__hero" aria-labelledby="fatqat-home-title" markdown>

<div class="fatqat-home__hero-copy" markdown>

<p class="fatqat-home__eyebrow">Quantum SDK</p>

# One Program. Three levels of physical detail. { #fatqat-home-title }

<p class="fatqat-home__summary" markdown>
Write a quantum computation once, as a [`Program`][fatqat.Program]. Then study
logical behavior, test device constraints, or follow time-dependent physical
dynamics without changing authoring models.
</p>

<div class="fatqat-home__actions" markdown>
[Build your first Program](guide/quickstart.md){ .md-button .md-button--primary }
[Compare execution levels](guide/execution-models.md){ .md-button }
</div>

<p class="fatqat-home__requirements" markdown>
Python 3.12+ · [Install from source](guide/quickstart.md)
</p>

</div>

<div class="fatqat-home__model" role="img" aria-label="One FatQat Program runs through general simulation, a hardware profile, or Hamiltonian emulation, with each path returning a Job and Result.">
<div class="fatqat-home__model-node"><code>Program</code></div>
<div class="fatqat-home__model-connector" aria-hidden="true"></div>
<div class="fatqat-home__model-choice">choose one execution level</div>
<div class="fatqat-home__model-targets">
<div class="fatqat-home__model-target">
<strong>General simulation</strong>
<span>states · counts · noise</span>
</div>
<div class="fatqat-home__model-target">
<strong>Hardware profile</strong>
<span>native gates · topology</span>
</div>
<div class="fatqat-home__model-target">
<strong>Hamiltonian emulation</strong>
<span>pulses · leakage · dynamics</span>
</div>
</div>
<div class="fatqat-home__model-connector" aria-hidden="true"></div>
<div class="fatqat-home__model-node"><code>Job</code> → <code>Result</code></div>
</div>

</section>

<div class="grid cards fatqat-home__benefits" markdown>

-   :material-vector-combine: **One authoring object**

    Keep gates, measurements, conditions, parameters, qudits, and physical
    controls in one Program.

-   :material-tune-variant: **Choose the fidelity**

    Model only the physical detail that the question actually requires.

-   :material-swap-horizontal-bold: **Keep the workflow**

    Submit work and inspect outputs through the same `Job` and `Result` concepts.

</div>

## See the complete workflow

<div class="grid fatqat-home__workflow" markdown>

<div markdown>

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

</div>

<div class="fatqat-home__workflow-result" markdown>

![A bar chart containing only the correlated 00 and 11 Bell-state outcomes.](assets/generated/guide/quickstart-counts.png)

<p class="fatqat-home__workflow-caption">One thousand seeded shots return only correlated outcomes.</p>

</div>

</div>

This Bell Program contains the complete circuit-level workflow. The same
Program abstraction can also carry reusable parameters, mixed local
dimensions, classical conditions, and direct physical controls.

## Choose where to continue

<div class="grid cards fatqat-home__destinations" markdown>

-   :material-play-circle-outline: **[Quickstart](guide/quickstart.md)**

    Build, draw, and run a first Program.

-   :material-book-open-page-variant-outline: **[User guide](guide/index.md)**

    Learn concepts and complete workflows.

-   :material-flask-outline: **[Tutorials](tutorials/index.md)**

    Explore executable algorithm and physics studies.

-   :material-format-list-bulleted: **[API reference](api/index.md)**

    Find exact signatures and validation contracts.

</div>

</div>
