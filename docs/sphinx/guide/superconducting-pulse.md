# Superconducting pulse simulation

`fatqat.backends.PulseBackend` simulates calibrated superconducting native
`RX`, `RY`, virtual `RZ`, `iSwap`, and oriented `CZ` operations in physical
three-level transmon space.  Build its immutable model and calibration from
ordinary JSON data; QuTiP and qutip-qip objects are private implementation
details and never appear in inputs or results.

```python
import fatqat as fq

model = fq.backends.load_physics_model(model_document)
calibration = fq.backends.load_calibration_spec(calibration_document, model)
backend = fq.backends.PulseBackend(model, calibration)

program = fq.Program(1)
program.add(fq.ops.RX(0.4), 0)
result = backend.run(
    program, result_config={"counts": False, "final_state": True}
).result()
rho = result.get_density_matrix()  # NumPy, shape (3**n, 3**n)
```

The frontend remains qubit-only, embedded into levels `|0>` and `|1>` of
each transmon.  Measurement is physical qutrit collapse followed by the
reported-bit mapping `0 -> 0`, `1 -> 1`, `2 -> 1`; optional readout confusion
then acts on that reported bit.  Dynamic runs replay serially so a later
classical guard reads the reported (and, if enabled, confused) value.

Only the listed native operations and model-declared coupling edges are
supported.  Matrix-family fake superconducting backends remain distinct
two-level targets.
