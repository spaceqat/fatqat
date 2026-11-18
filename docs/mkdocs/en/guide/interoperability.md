# Connect FatQat with OpenQASM and Qiskit { #bring-programs-to-and-from-openqasm-and-qiskit }

If you already have a quantum algorithm written in OpenQASM 2 or 3, or as a
Qiskit circuit, you can bring the circuit into FatQat to run it and analyse the
results. To take a supported FatQat Program to another platform, export it in
the OpenQASM version that the other tool expects.

The route depends on where your circuit starts and which tools you want to
use to run it.

- **OpenQASM 2 or 3 source.** Import it into FatQat and run it. You can also export a Program in either version.
- **A Qiskit circuit.** Convert it to a [`Program`][fatqat.Program] to run it and analyse the result in FatQat.
- **A Qiskit application.** Use FatQat as a Qiskit backend and keep the Qiskit job and result APIs.

## Import, run, and export OpenQASM

FatQat's built-in `from_qasm` reads both OpenQASM 2 and 3. This example uses
OpenQASM 3 to describe the measured Bell circuit from the quickstart.

```pycon
>>> import fatqat as fq
>>> from fatqat.qasm import from_qasm, to_qasm
>>> source = """
... OPENQASM 3.0;
... include "stdgates.inc";
... qubit[2] q;
... bit[2] c;
... h q[0];
... cx q[0], q[1];
... c = measure q;
... """
>>> program = from_qasm(source)
>>> isinstance(program, fq.Program)
True
```

The imported value is an ordinary Program. Choose a backend, run it, and read
the normal FatQat result:

```pycon
>>> counts = (
...     fq.simulator.Simulator()
...     .run(program, shots=100, execution_config={"seed": 7})
...     .result()
...     .get_counts()
... )
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
100
```

Use `to_qasm` to export a supported Program. It writes OpenQASM 3 by default.
For a tool that expects OpenQASM 2, pass `version=2`.

```pycon
>>> exported = to_qasm(program)
>>> exported.splitlines()[0]
'OPENQASM 3.0;'
>>> "cx q[0], q[1];" in exported
True
>>> exported_v2 = to_qasm(program, version=2)
>>> exported_v2.splitlines()[0]
'OPENQASM 2.0;'
>>> isinstance(from_qasm(exported_v2), fq.Program)
True
```

An export is a normalized representation, not a promise to reproduce the
original text byte for byte. Register names may be made safe or unique, whole
register operations may be expanded, and Program metadata is not part of
OpenQASM.

OpenQASM exchange currently represents bound, dimension-two circuit Programs.
Unsupported language constructs or Program features fail explicitly instead
of being approximated. See the [OpenQASM API reference](../api/interoperability/openqasm.md) for supported statements, version-specific
conditions, file input, and conversion errors.

## Convert a Qiskit circuit into a Program

Install Qiskit in the same Python environment as FatQat for both Qiskit examples.

```bash
python -m pip install qiskit
```

Create one measured Bell circuit and use it in both examples.

<!-- guide-test: requires qiskit -->
```python
import fatqat as fq
from qiskit import QuantumCircuit

bell_circuit = QuantumCircuit(2, 2, name="bell")
bell_circuit.h(0)
bell_circuit.cx(0, 1)
bell_circuit.measure([0, 1], [0, 1])
```

To use FatQat's execution and result APIs, pass the circuit to
[`circuit_to_program`][fatqat.qiskit.circuit_to_program] and run the returned Program.

<!-- guide-test: requires qiskit -->
```pycon
>>> from fatqat.qiskit import circuit_to_program
>>> bell_program = circuit_to_program(bell_circuit)
>>> fatqat_result = (
...     fq.simulator.Simulator()
...     .run(bell_program, shots=100, execution_config={"seed": 7})
...     .result()
... )
>>> print(dict(sorted(fatqat_result.get_counts().items())))
{'00': 50, '11': 50}
```

## Use FatQat as a Qiskit backend

To keep the Qiskit job and result APIs, pass
[`FatqatBackend`][fatqat.qiskit.FatqatBackend] to Qiskit's transpiler.
Then submit the same Bell circuit after transpilation.

<!-- guide-test: requires qiskit -->
```pycon
>>> from qiskit import generate_preset_pass_manager
>>> from fatqat.qiskit import FatqatBackend
>>> backend = FatqatBackend()
>>> pass_manager = generate_preset_pass_manager(
...     backend=backend,
...     optimization_level=1,
...     seed_transpiler=7,
... )
>>> compatible_circuit = pass_manager.run(bell_circuit)
>>> job = backend.run(compatible_circuit, shots=100, seed_simulator=7)
>>> qiskit_result = job.result()
>>> print(dict(sorted(qiskit_result.get_counts().items())))
{'00': 50, '11': 50}
```

With the seeds shown, both routes return 50 counts for `"00"` and 50 for
`"11"`. The direct conversion gives a FatQat `Result` with access to the data
your run produced. The backend route gives a Qiskit job and result, with
counts formatted using Qiskit's conventions.

For asymmetric circuits, check bit order when comparing results. FatQat
puts slot 0 on the left in count strings. Qiskit puts bit 0 on the right
within a register. When comparing arrays or Pauli labels, use the conversions
in the [Qiskit API reference](../api/interoperability/qiskit.md).

The adapter accepts static circuits with bound parameters. Transpile each
circuit for the backend's target, and use a FatQat `NoiseModel` for noisy runs.
The [Qiskit API reference](../api/interoperability/qiskit.md) describes
supported instructions, noise, run options, and result formats.
