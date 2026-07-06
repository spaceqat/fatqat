# Quickstart

Install fatqcat into your environment (editable, from a checkout):

```sh
pip install -e .
```

Build a program, run it, and read back the measurement counts:

```python
import fatqcat as fqc

program = fqc.Program(2, 2)          # 2 qubits, 2 clbits
program.add(fqc.ops.H, 0)
program.add(fqc.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

result = fqc.backends.StateVectorBackend().run(program, shots=1000).result()
print(result.get_counts())          # e.g. {"00": 512, "11": 488}
```

That's the whole shape of every fatqcat program: build a
{py:class}`~fatqcat.Program`, add gates and measurements, run it on a backend,
read the {py:class}`~fatqcat.Result`. The rest of this guide fills in the
details — see [Concepts](concepts.md) for the mental model behind these four
objects.
