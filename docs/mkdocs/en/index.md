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

<div class="grid cards" markdown>

-   :material-play-circle-outline: **[Run your first Program](guide/quickstart.md)**

    Build and draw the Bell circuit, run it, and turn its counts into a figure.

-   :material-transit-connection-variant: **[Learn the Program](guide/program.md)**

    Add registers, measurements, conditions, parameters, qudits, and mixed local
    dimensions without changing authoring models.

-   :material-source-branch: **[Choose the modeling level](guide/execution-models.md)**

    Compare general simulation, a hardware profile, and Hamiltonian-level
    emulation using one Program.

-   :material-sine-wave: **[Study hardware behavior](guide/hardware-profile-simulation.md)**

    Work with native gates, layouts, connectivity, pulse controls, leakage, and
    physical models.

-   :material-flask-outline: **[Work through tutorials](tutorials/index.md)**

    Continue into complete algorithm and physics case studies with downloadable
    sources.

-   :material-format-list-bulleted: **[Look up the API](api/index.md)**

    Find exact signatures, supported operations, shapes, units, and validation
    rules.

</div>
