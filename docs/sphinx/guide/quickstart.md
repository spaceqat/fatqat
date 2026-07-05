# Quickstart

Install qnsim into your environment (editable, from a checkout):

```sh
pip install -e .
```

Build a program, run it, and read back the measurement counts:

```python
import qnsim as qs

program = qs.Program(2, 2)          # 2 qubits, 2 clbits
program.add(qs.ops.H, 0)
program.add(qs.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

result = qs.backends.StateVectorBackend().run(program, shots=1000).result()
print(result.get_counts())          # e.g. {"00": 512, "11": 488}
```

That's the whole shape of every qnsim program: build a
{py:class}`~qnsim.Program`, add gates and measurements, run it on a backend,
read the {py:class}`~qnsim.Result`. The rest of this guide fills in the
details — see [Concepts](concepts.md) for the mental model behind these four
objects.
