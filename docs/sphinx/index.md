# qnsim

qnsim is a quantum simulator: build a {py:class}`~qnsim.Program` out of
registers, gates, and measurements, run it on a backend, and read back
counts or a statevector.

```python
import qnsim as qs

program = qs.Program(2, 2)          # 2 qubits, 2 clbits
program.add(qs.ops.H, 0)
program.add(qs.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

result = qs.backends.StateVectorBackend().run(program, shots=1000).result()
print(result.get_counts())          # e.g. {"00": 512, "11": 488}
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book` User guide
:link: guide/index
:link-type: doc

Task-oriented pages: quickstart, the core concepts, the gate catalogue,
measurement/conditions, and running programs to read back results.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Autodoc-generated reference for every public class, function, and gate,
grouped by namespace (``qs``, ``qs.ops``, ``qs.backends``, ``qs.errors``).
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

guide/index
api/index
```
