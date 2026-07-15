# fatqat

fatqat is a quantum noisy simulator (MVP Phase 1). Build a `Program` out of
registers, gates, and measurements, run it on the simulator backend, and read
back counts, a statevector, or a density matrix.

```python
import fatqat as fq

program = fq.Program(2, 2)          # 2 qubits, 2 clbits
program.add(fq.ops.H, 0)
program.add(fq.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

backend = fq.backends.SimulatorBackend(method="SV")   # or method="DM"
result = backend.run(program, shots=1000).result()
print(result.get_counts())          # e.g. {"00": 512, "11": 488}
```

The `method` selects the state representation: `"SV"`/`"statevector"` for
pure-state simulation, `"DM"`/`"density_matrix"` for exact mixed-state
simulation (e.g. reset acting on entangled qubits).

## Dev setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync              # install runtime + dev dependencies into .venv
uv run pytest        # run the test suite
```

## Documentation

The user guide and API reference live in `docs/sphinx`. Build them locally
with warnings-as-errors, so a missing docstring or broken cross-reference
fails the build instead of silently vanishing:

```sh
uv sync --group docs
uv run sphinx-build -b html -W docs/sphinx docs/sphinx/_build
```

Then open `docs/sphinx/_build/index.html`. There's no CI job or hosting for
this yet — it's for local/internal use.

## Project layout

- `src/fatqat/` — package source: `Program`, `operations` (gates,
  measurement, reset), `backends` (statevector, parallel), `implementation`
  (matrix backend), registers, jobs, results, errors.
- `tests/` — pytest suite.
- `docs/sphinx/` — user guide and API reference (see above).
