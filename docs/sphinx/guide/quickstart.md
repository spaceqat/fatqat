# Quickstart

Install fatqat into your environment (editable, from a checkout):

```sh
pip install -e .
```

Build a program, run it, and read back the measurement counts:

```python
import fatqat as fc

program = fc.Program(2, 2)          # 2 qubits, 2 clbits
program.add(fc.ops.H, 0)
program.add(fc.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

result = fc.backends.StateVectorBackend().run(program, shots=1000).result()
print(result.get_counts())          # e.g. {"00": 512, "11": 488}
```

That's the whole shape of every fatqat program: build a
{py:class}`~fatqat.Program`, add gates and measurements, run it on a backend,
read the {py:class}`~fatqat.Result`. The rest of this guide fills in the
details — see [Concepts](concepts.md) for the mental model behind these four
objects.
