# Bring Programs to and from OpenQASM and Qiskit

Already have a circuit? FatQat can import OpenQASM 3, convert a Qiskit
`QuantumCircuit` into a {py:class}`~fatqat.Program`, or appear inside Qiskit as
a backend.

## Pick an integration path

- Use OpenQASM when the exchange artifact should be portable source text.
- Convert a Qiskit circuit when FatQat should own execution and interpretation.
- Use `FatqatBackend` when Qiskit should continue to own the surrounding job
  and result workflow.

## Import, run, and export OpenQASM

The OpenQASM translator is built into FatQat and does not require Qiskit. This
OpenQASM 3 source describes the same measured Bell circuit used in the
quickstart:

```{doctest}
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

```{doctest}
>>> counts = (
...     fq.simulator.Simulator()
...     .run(program, shots=100, simulation_config={"seed": 7})
...     .result()
...     .get_counts()
... )
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
100
```

Export a supported Program to OpenQASM 3 with `to_qasm`:

```{doctest}
>>> exported = to_qasm(program)
>>> exported.splitlines()[0]
'OPENQASM 3.0;'
>>> "cx q[0], q[1];" in exported
True
```

An export is a normalized representation, not a promise to reproduce the
original text byte for byte. Register names may be made safe or unique, whole
register operations may be expanded, and Program metadata is not part of
OpenQASM.

OpenQASM exchange currently represents bound, dimension-two circuit Programs.
Unsupported language constructs or Program features fail explicitly instead
of being approximated. See the {doc}`OpenQASM API reference
<../api/interoperability/openqasm>` for supported statements, version-specific
conditions, file input, and conversion errors.

## Convert a Qiskit circuit into a Program

Qiskit integration is optional. Install Qiskit in the same environment as
FatQat; Qiskit Aer is not required:

```bash
python -m pip install qiskit
```

Use {py:func}`~fatqat.qiskit.circuit_to_program` when you want to cross the
boundary into FatQat and then use its backends and result model:

```python
from qiskit import QuantumCircuit

import fatqat as fq
from fatqat.qiskit import circuit_to_program

circuit = QuantumCircuit(2, 2, name="bell")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

program = circuit_to_program(circuit)
result = (
    fq.simulator.Simulator()
    .run(program, shots=100, simulation_config={"seed": 7})
    .result()
)
counts = result.get_counts()
```

From `program` onward this is the same workflow as a Program authored directly
in Python. In particular, you can choose a FatQat execution model and work
with FatQat's `Result` accessors.

## Use FatQat as a Qiskit backend

Use {py:class}`~fatqat.qiskit.FatqatBackend` instead when the surrounding
application should remain a Qiskit workflow. Transpile to the backend's target
before submitting the circuit:

```python
from qiskit import QuantumCircuit, generate_preset_pass_manager
from fatqat.qiskit import FatqatBackend

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

backend = FatqatBackend()
pass_manager = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1,
)
compatible_circuit = pass_manager.run(circuit)

job = backend.run(compatible_circuit, shots=100, seed_simulator=7)
qiskit_result = job.result()
counts = qiskit_result.get_counts()
```

This path returns Qiskit's job and result types, with Qiskit-formatted counts.
Use direct conversion when you need FatQat state, map, or observable results;
the Qiskit backend adapter is intentionally counts-oriented.

:::{note}
The adapter accepts bound, static circuits in its advertised target basis.
Transpilation handles other supported Qiskit gate forms; unsupported dynamic
control flow is rejected. A `FatqatBackend` accepts a FatQat `NoiseModel`, not
a Qiskit Aer noise model.
:::

The {doc}`Qiskit API reference <../api/interoperability/qiskit>` documents the
exact target, conversion behavior, run options, batches, memory output, and
error types.

When an input cannot be represented by the supported Program subset, the
converter reports an error rather than guessing at its meaning.
