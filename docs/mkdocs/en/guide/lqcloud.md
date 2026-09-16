# Run a Program on LQCloud hardware

The optional LQCloud adapter sends a FatQat [`Program`][fatqat.Program] to
the 17-qubit `QZ01-surface_code` processor, also called QEC17. You write the
circuit in the processor's native gate set, choose physical qubits, and use
the LQCloud SDK to follow the submitted job and read its result.

The adapter translates native instructions and validates the placement. It
does not decompose gates or route a circuit. Start with the
[supported operations](../api/interoperability/lqcloud.md#supported-programs)
when preparing a program for this target.

## Install the SDK

Install the optional SDK alongside FatQat:

```sh
python -m pip install lqcloud==0.4.2
```

You need an LQCloud account with access to `QZ01-surface_code` to submit a
job. Keep the API key outside your source code. The submission example below
reads it from the `LQCLOUD_API_KEY` environment variable.

## Prepare a native circuit

This two-qubit circuit prepares a Bell state using native half rotations and
`CZ`:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.YHalf, 0)
program.add(ops.MYHalf, 1)
program.add(ops.CZ, (0, 1))
program.add(ops.YHalf, 1)
```

Do not add measurements or a reset to this program. The adapter appends a
barrier and measurement of every program qubit; the platform handles initial
state preparation. Classical conditions and dynamic circuits are unsupported.

The general [`Simulator`][fatqat.simulator.Simulator] supports these gates,
so you can check the ideal state locally before submitting:

```python
state = fq.simulator.Simulator().run(program).result().get_statevector()
```

This checks the circuit's ideal behavior; it does not model QEC17 device noise.
You can also inspect the translated SDK circuit without connecting to LQCloud:

```python
from fatqat.lqcloud import program_to_qec17_circuit

circuit = program_to_qec17_circuit(program)
print(circuit.draw())
```

Translation alone does not validate a physical layout. That check happens when
the backend runs the program.

## Choose a placement and submit

`initial_layout[k]` gives the physical qubit for logical qubit `k`. For multiple
quantum registers, logical indices follow register declaration order. Here
logical qubits 0 and 1 use physical qubits 6 and 11, which share a QEC17 `CZ`
connection. See the [coupling graph](../api/interoperability/lqcloud.md#placement)
when choosing a larger layout.

Run the following code when you intend to submit a real hardware job:

```python
import os
from fatqat.lqcloud import LQCloudQEC17Backend

backend = LQCloudQEC17Backend(api_key=os.environ["LQCLOUD_API_KEY"])
job = backend.run(program, shots=1024, initial_layout=[6, 11])
```

The adapter checks each `CZ` after applying your layout and rejects
unconnected pairs. It also checks qubit capacity, dimensions, layout indices,
and the native instruction set. It never inserts routing gates. Omitting
`initial_layout` uses the identity mapping for validation, so an explicit layout
is usually useful for circuits with two-qubit gates.

## Follow the job and read counts

`run()` returns the SDK's asynchronous job. Use the LQCloud job methods to
check its status, cancel it, or wait for the result:

```python
print(job.status())
result = job.result()
counts = result.get_counts()
print(counts)
```

These are LQCloud job and result objects. Their status values, waiting behavior,
result formats, and bitstring ordering follow the SDK, rather than FatQat's
local [`Job`][fatqat.Job] and [`Result`][fatqat.Result] APIs. Do not assume the
cloud bitstrings use FatQat's ordering when comparing results.

Additional keyword arguments to `run()`, such as `readout_correction` and
`result_format`, go directly to the SDK. Leave `result_format` unset to use the
SDK's default. The [LQCloud API reference](../api/interoperability/lqcloud.md)
lists the exact input constraints and translation behavior.
