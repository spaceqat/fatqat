# Troubleshooting

These checks address the most common first-run problems without requiring
you to inspect backend or engine internals.

| Symptom | What to check |
| --- | --- |
| `import fatqat` fails | Activate the virtual environment where you ran `python -m pip install -e .`, then rerun the command. |
| Counts are unavailable | Add a measurement and request `{"counts": True}` when the output must be explicit. |
| A state accessor is unavailable | Request the corresponding result field and use the backend method that provides it: `statevector` for `get_statevector()` or `density_matrix` for `get_density_matrix()`. |
| An operation is unsupported | Start with {py:class}`~fatqat.simulator.Simulator`. A constrained target such as {py:class}`~fatqat.simulator.SCQubitIBMSimulator` or {py:class}`~fatqat.simulator.AtomArraySimulator` intentionally accepts only its own native gate set and connectivity. |
| An integer target is ambiguous | The program has multiple registers. Pass a reference such as `program.quantum_registers[1][0]` instead of a bare integer. |
| Counts vary between runs | Measurement is sampled. Use more `shots` for a steadier estimate and pass `seed=` for reproducible samples. |

## Result fields

A {py:class}`~fatqat.Result` contains only the fields produced by that run. Before reading an
optional field, inspect `result.available_data`:

```python
if "statevector" in result.available_data:
    state = result.get_statevector()
```

For a measurement, reset, or noisy statevector program, a single final
state may be stochastic. Request a statevector only with `shots=1` in that
case, or use a density matrix when you need an exact noisy state.

## Count order

Count strings are little-endian: clbit 0 is on the right. Read the
[results guide](running-and-results.md) before using counts to compare with
another framework.

## Still blocked?

Reduce the program to a short example: create a {py:class}`~fatqat.Program`, add one gate,
add a measurement, and run {py:class}`~fatqat.simulator.Simulator`. Then add operations one at
a time. This normally identifies whether the issue is program construction,
a requested result field, or a constrained backend choice.
