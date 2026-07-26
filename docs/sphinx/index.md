# fatqat

fatqat is a Python quantum simulator. Build a `Program` from registers,
gates, and measurements; run that program on a backend; then read the
requested data from a `Result`.

The normal workflow stays at that level. You build programs and use
backends; the simulator engine and its execution machinery stay behind the
backend boundary.

## Your first program

```python
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2, 2)  # two qubits and two classical bits
program.add(op.H, 0)
program.add(op.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

backend = fq.backends.SimulatorBackend()
job = backend.run(program, shots=1000)
result = job.result()
print(result.get_counts())
```

This prepares an entangled Bell state. The exact counts vary from run to
run, but only `"00"` and `"11"` should occur.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`play` Get started
:link: guide/quickstart
:link-type: doc

Install fatqat from a checkout and run the complete example above.
:::

:::{grid-item-card} {octicon}`book` Learn the model
:link: guide/concepts
:link-type: doc

Understand programs, registers, operations, backends, and results before
moving to optional features.
:::

:::{grid-item-card} {octicon}`graph` Read results
:link: guide/running-and-results
:link-type: doc

Choose counts, a statevector, or a density matrix and interpret bit order.
:::

:::{grid-item-card} {octicon}`list-unordered` Supported API
:link: api/index
:link-type: doc

Look up the application-facing objects used in the guide.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

guide/index
api/index
```
