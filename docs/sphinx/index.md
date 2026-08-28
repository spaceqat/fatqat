# FatQat

FatQat lets you write a quantum computation once, as a `Program`, and study it
at the level your question needs: logical circuit behavior, hardware-profile
constraints, or time-dependent physical dynamics.

Here is the complete circuit-level workflow:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure((0, 1), (0, 1))

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()

print(result.get_counts())
```

This prepares a Bell state. The sample changes with the seed, but the only
possible outcomes are `"00"` and `"11"`. The same `Program` abstraction also
carries parameters, qudits, classical conditions, and direct physical
controls.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`play` Run your first Program
:link: guide/quickstart
:link-type: doc

Build and draw the Bell circuit, run it, and turn its counts into a figure.
:::

:::{grid-item-card} {octicon}`workflow` Learn the Program
:link: guide/program
:link-type: doc

Add registers, measurements, conditions, parameters, qudits, and mixed local
dimensions without changing authoring models.
:::

:::{grid-item-card} {octicon}`git-branch` Choose the modeling level
:link: guide/execution-models
:link-type: doc

See one Program run through general simulation, a hardware profile, and a
Hamiltonian-level emulator.
:::

:::{grid-item-card} {octicon}`pulse` Study hardware behavior
:link: guide/hardware-profile-simulation
:link-type: doc

Work with native gates, layouts, connectivity, pulse controls, leakage, and
physical models.
:::

:::{grid-item-card} {octicon}`beaker` Work through tutorials
:link: tutorials/index
:link-type: doc

Continue into systematic algorithm and physics case studies, with downloadable
notebooks.
:::

:::{grid-item-card} {octicon}`list-unordered` Look up the API
:link: api/index
:link-type: doc

Find exact signatures, supported operations, shapes, units, and validation
rules.
:::

::::

```{toctree}
:maxdepth: 3
:hidden:

guide/index
tutorials/index
api/index
```
