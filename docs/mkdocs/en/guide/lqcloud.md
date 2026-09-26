# Run a circuit on LogicalQubit Cloud

Compile a `LogicalProgram` or OpenQASM circuit with FatQat, then run it on
`AGate-100`, `QZ01-surface_code`, or `MQ02`. FatQat decomposes logical gates,
maps qubits to the selected device, and routes interactions using SABRE before
preparing the cloud circuit.

## Install and connect

Use **Python 3.12** for this integration. The supported LQCloud SDK version is
0.5.0; its Python requirement is narrower than FatQat's core requirement.

```sh
python -m pip install 'fatqat[lqcloud]'
```

Set `LQCLOUD_API_KEY` in your environment using the key from your cloud
account. Do not put the key in a notebook or commit it with a script. The SDK
can also read an existing saved account configuration.

```python
import fatqat as fq
from fatqat.lqcloud import LQCloudBackend

backend = LQCloudBackend("MQ02")
```

Construction loads the device's capacity and coupling graph without submitting
a task. To change devices, construct another backend with its exact name and
compile for that backend.

## Compile and submit

```python
circuit = fq.LogicalProgram(3, 3)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.add(fq.operations.CCX, (0, 1, 2))
circuit.measure_all()

compiled = fq.compiler.compile_to_sc(circuit, backend, seed=7)
```

The compiled result is ready for `backend.run()`. Its `.output` contains the
physical native instructions and initial/final layouts; `.route` lists the
compiler stages. Computational instructions use H, RZ and CZ. FatQat also
decomposes iSWAP, SWAP, CY, CS, CPhase and CSwap. No manual conversion or
bridge call is needed.

**The next call submits a hardware task and can incur charges.** Check the
device's current status and price in the
[cloud console](https://cloud.logicalqubit.com/console) before running it.

```python
job = backend.run(compiled, shots=1024)
print(job.job_id)
result = job.result(timeout=300)
print(result.get_counts())
```

The result is a FatQat `Result` containing shot counts, not a statevector.
Count keys use source classical-register order with classical slot 0 on the
left, including zero-filled unwritten slots. Device noise means counts need
not match ideal simulation exactly.

## Compile OpenQASM

Use the same backend with the QASM entry point:

```python
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
ccx q[0], q[1], q[2];
measure q -> c;
"""
compiled = fq.compiler.compile_qasm_to_sc(qasm, backend, seed=7)
# Submit when ready:
# job = backend.run(compiled, shots=1024)
```

OpenQASM 2 does not define `iswap` in `qelib1.inc`. Include an explicit gate
definition, or export a FatQat circuit with `fq.qasm.to_qasm(circuit)`, which
supplies that definition.

## Follow a task

`job.status()` queries PENDING, QUEUED, RUNNING, COMPLETED, FAILED or CANCELLED.
`job.result(timeout=300)` waits using SDK polling. A `lqcloud.JobTimeoutError`
leaves the task running: call `result()` again on the same handle later.
`timeout=None` waits without a deadline; otherwise use a positive integer
number of seconds.

`job.cancel()` requests cancellation. A task already executing on the QPU may
not be cancellable; the SDK reports this through `lqcloud.JobError`.

## Measurements

Use static qubit circuits with numeric parameters and terminal measurements.
Partial measurements and multiple classical registers are supported. Bind
symbolic parameters before compiling, just as for the simulator route.
Compilation without measurements is useful for inspecting native output;
submission requires at least one measurement and never adds `measure_all()`.

The adapter prepares the barrier before readout. The official SDK supplies
device initialization and the appropriate readout protocol, including X21
preparation for measure2-only devices. Users keep writing ordinary
`measure()` or `measure_all()` calls. Tasks use raw sampling with readout
correction and automatic dynamic decoupling disabled.

See the [LQCloud API reference](../api/interoperability/lqcloud.md) for
constructor options, result fields and exceptions.
