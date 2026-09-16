# FatQat × LQCloud QEC17 Interface

This directory contains the tests and a directly runnable example for FatQat's lightweight integration with the LQCloud `QZ01-surface_code` (QEC17) processor. The integration is a **native circuit-language translator and submission wrapper**, not a compiler: users write a FatQat `Program` using the QEC17 native gate set, and FatQat translates each operation into an LQCloud `QuantumCircuit` before submission.

## Implemented functionality

- `program_to_qec17_circuit(program)` translates FatQat native gates into the corresponding LQCloud native gates.
- `LQCloudQEC17Backend(api_key).run(...)` connects to the fixed `QZ01-surface_code` backend and returns the native asynchronous LQCloud `Job`.
- The adapter supports QEC17 native single-qubit gates, arbitrary-angle `RZ`, `SU2`, and the two-qubit `CZ` gate.
- `initial_layout` maps logical qubits to the 17 physical qubits. Every mapped `CZ` edge is checked against the QEC17 coupling graph before submission.
- A global `barrier()` and `measure_all()` are appended automatically at the end of the circuit.
- User-supplied `Reset`, measurement instructions, non-native gates, and dynamic circuits are rejected explicitly. Initial-state preparation is handled by LQCloud, so users must not add a manual reset.
- Conversion leaves the user's `Program` unchanged. Only an `I` gate explicitly written by the user is translated into `qc.id()`; the adapter inserts no idle operations.

The current integration does not perform gate decomposition, optimization, routing, or dynamic-circuit execution. Input circuits must already use the QEC17 native gate set.

## Four-qubit GHZ example

[`run_qec17_four_qubit_ghz.py`](run_qec17_four_qubit_ghz.py) uses the valid QEC17 physical path `6—11—4—12` to prepare and measure

```text
(|0000> + |1111>) / sqrt(2)
```

Install the optional SDK:

```powershell
python -m pip install lqcloud==0.4.2
```

Set the API key and run the example:

```powershell
$env:LQCLOUD_API_KEY="your real API key"
python tests/lqcloud/run_qec17_four_qubit_ghz.py
```

Use the environment where FatQat and the SDK are installed. The example submits
a real hardware job when run; never commit a real API key to the repository.

## Tests

[`test_qec17_backend.py`](test_qec17_backend.py) covers gate translation, terminal barrier and measurement insertion, invalid inputs, layout and topology validation, backend option forwarding, and offline serialization with the real SDK. The tests do not connect to LQCloud or submit jobs to real hardware.

```powershell
python -m pytest tests/lqcloud
```

The optional `lqcloud` dependency group is included in `test-full` so the full
test environment also runs the real SDK serialization checks. Neither the
tests nor circuit conversion require an API key.
