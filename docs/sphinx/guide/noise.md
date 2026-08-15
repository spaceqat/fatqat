# Noise

Noise lives in a {py:class}`~fatqat.NoiseModel`, a container built separately
from the program and passed to the backend at construction. The program never
changes: the same circuit runs ideal or noisy depending only on which backend
runs it.

```python
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2, 2)
program.add(op.H, 0)
program.add(op.CX, (0, 1))
program.measure((0, 1), (0, 1))

noise = fq.NoiseModel()
noise.add_channel(fq.noise.Depolarizing(p=0.05), operation=op.CX)

ideal = fq.simulator.Simulator(method="DM")
noisy = fq.simulator.Simulator(method="DM", noise=noise)
print(ideal.run(program, shots=1000, simulation_config={"seed": 7}).result().get_counts())
print(noisy.run(program, shots=1000, simulation_config={"seed": 7}).result().get_counts())
```

Two kinds of noise exist, with deliberately different mechanics:

- **Quantum channels** ({py:meth}`~fatqat.NoiseModel.add_channel`) act on the
  quantum state. Their optional `operation` selector determines whether they
  are always active or scoped to matching operation occurrences.
- **Readout error** ({py:meth}`~fatqat.NoiseModel.add_readout_error`) is
  classical: the physical collapse always keeps the true outcome, and only
  the *reported* measurement value is resampled.

## Channels

A channel descriptor holds physical parameters only — never Kraus arrays —
the same way a gate like `RX(0.3)` never stores its matrix. The built-in
catalog:

- {py:class}`~fatqat.noise.Depolarizing` `(p)` — `rho -> (1-p) rho + p I/d`.
  Acts jointly on all subsystems of the gate it is attached to, so the same
  descriptor depolarizes a single qubit, a qutrit, or the joint space of a
  two-qubit gate.
- {py:class}`~fatqat.noise.AmplitudeDamping` `(p or rate)` — energy relaxation
  toward the ground state, one value per adjacent-level transition (a qubit
  takes `p=(p,)` or `rate=(rate,)`). Single-subsystem.
- {py:class}`~fatqat.noise.PhaseDamping` `(p or rate)` — pure dephasing:
  populations are untouched, coherences decay (at factor `1-p` for a qubit).
  Single-subsystem.

Attaching several channels to the same gate stacks them as independent
mechanisms, applied in registration order:

```python
noise = fq.NoiseModel()
noise.add_channel(fq.noise.AmplitudeDamping(p=(0.01,)), operation=op.H)
noise.add_channel(fq.noise.PhaseDamping(p=0.02), operation=op.H)
```

Channels are dimension-generic: attached to a gate on a qutrit register,
`Depolarizing` and `PhaseDamping` resolve at `d=3` automatically, and
`AmplitudeDamping` takes `d-1` ladder values (`p=(p10, p21)`).

### Probability versus rate

`AmplitudeDamping` and `PhaseDamping` accept exactly one of `p` (a keyword-only,
mutually exclusive pair with `rate`):

- `p` describes one finite channel application: the probability the transition
  happens once, given the gate actually ran.
- `rate` describes a continuous generator, in the inverse of whatever time
  unit the target backend declares. Converting between the two always
  requires a duration - `p(t) = 1 - exp(-rate * t)` - and neither the
  descriptor nor `NoiseModel` ever infers or stores a time unit; that
  responsibility belongs to the backend.

Both backend families accept the same descriptor, through different
implementation maps:

- `Simulator` (the matrix family) resolves `p` directly into Kraus
  operators, and rejects `rate` mode - no gate carries a duration in matrix
  lowering today, so there is nothing to convert it with.
- A pulse emulator whose effective `LindbladImplementationMap` registers the
  descriptor resolves either mode into a collapse-operator rate, using the
  realized operation's own duration. Without an `operation` selector, rate
  mode is always active, including idle intervals. Family defaults differ;
  supplied maps replace those defaults.

A probability of exactly `1` is a valid finite channel but has no finite
rate, so converting it to `rate` mode raises rather than silently
approximating. A nonzero probability on a zero-duration operation is
likewise not convertible (there is no time over which the rate could act);
zero probability at zero duration is a well-defined no-op.

Registration scope is represented by `operation`: omitting it means
always-on, while `operation=op.X` selects matching `X` blocks. Rate mode does
not by itself mean globally active. Probability mode requires an operation
scope because a finite probability has no meaning without an interval.
Operation-scoped pulse resolution applies to eligible ordinary operations;
`NoiseModel` rejects direct `PulseOperation` as an operation selector.

## Targeting specific qubits

`add_channel(..., targets=...)` pins a channel to one target tuple. Two address
forms exist, and both resolve to the same flat indices at run time:

- **Program refs** — how a user pins noise to their own program's subsystems:

  ```python
  noise.add_channel(
      fq.noise.Depolarizing(p=0.1),
      operation=op.X,
      targets=(program.quantum_registers[0][0],),
  )
  ```

- **Flat subsystem indices** — how a device backend authors noise before any
  user program exists:

  ```python
  noise.add_channel(
      fq.noise.Depolarizing(p=0.1), operation=op.X, targets=(0,)
  )
  ```

The selection semantics, precisely:

- Without `targets`, a channel applies to *every* occurrence of the
  operation (the all-targets default).
- A specific-target entry replaces the all-targets default on the
  occurrences it matches, and only those. It can therefore *lower* the noise
  on its target by evicting a stronger default; restate the default at the
  specific level to keep it.
- Entries resolving to the same subsystems accumulate in registration order
  — an `int` selector and a ref selector landing on the same subsystem stack
  rather than override.

## Relaxation from T1/T2

{py:class}`~fatqat.noise.ThermalRelaxation` owns device T1/T2 timescales and
converts
an explicit gate duration into the equivalent channel pair — populations
decay by `1 - exp(-duration/t1)`, coherences by `exp(-duration/t2)` in total:

```python
relaxation = fq.noise.ThermalRelaxation(t1=60e-6, t2=80e-6)
damping, dephasing = relaxation.as_channels(duration=2e-6)
noise = fq.NoiseModel()
noise.add_channel(damping, operation=op.H)
noise.add_channel(dephasing, operation=op.H)
```

The library never derives durations itself — operations carry no time — so
the caller supplies how long the noisy gate takes. `t2 <= 2*t1` is enforced
(the physical bound: pure dephasing cannot be negative). On a pulse backend
with a registered `ThermalRelaxation` rule, register the descriptor with an
ordinary operation to scope it to realized blocks, or without an operation to
act throughout pulse and idle evolution:

```python
noise.add_channel(relaxation, targets=(program.quantum_registers[0][0],))
```

## Readout error

{py:meth}`~fatqat.NoiseModel.add_readout_error` takes a column-stochastic
confusion matrix `C[i, j] = P(report i | true j)` and an optional per-subsystem
target (ref or flat index; the default applies to every measured subsystem):

```python
import numpy as np

noise = fq.NoiseModel()
noise.add_readout_error(np.array([[0.98, 0.05],
                                  [0.02, 0.95]]))
```

Readout error is deliberately *not* a quantum channel. The collapse keeps
the true outcome and only the reported classical value is resampled, which
means:

- Qubit reuse after measurement evolves from the true post-measurement
  state, and a requested statevector/density matrix shows the true state.
- Feedforward conditions read the **reported** bit — what real control
  electronics see.
- Execution-strategy classification never changes: readout error rides the
  fast path untouched.

## How noise executes

The two simulation methods handle the same `NoiseModel` differently, and
converge to the same counts:

- `method="DM"` applies each channel exactly, as the Kraus sum
  `rho' = sum_i K_i rho K_i^H` — deterministic, so a noisy program still
  evolves once and samples counts at the end (the fast path).
- `method="SV"` cannot represent a mixed state, so each channel occurrence
  samples one Kraus branch per shot (quantum trajectories). Any channel
  therefore makes statevector execution stochastic and forces per-shot
  replay; exporting the statevector of a noisy program requires `shots=1`.

Prefer `method="DM"` for exact noisy distributions while the system fits in
memory (a density matrix is quadratically larger); prefer `method="SV"`
trajectories for larger systems or genuinely per-shot questions.

## Device-authored noise

A device backend can build its noise model from its own calibration facts —
the from-backend workflow. The fake superconducting target ships one:

```python
Fake = fq.simulator.SCQubitIBMSimulator
noisy_fake = Fake(noise=Fake.default_noise_model())
```

{py:meth}`~fatqat.simulator.SCQubitIBMSimulator.default_noise_model`
returns a fresh, ordinary `NoiseModel` (T1/T2 relaxation on `SX`, a `CZ`
depolarizing channel, asymmetric readout confusion; the virtual `RZ` stays
noise-free) — inspect it, extend it with your own channels, and pass it
back. The backend itself stays ideal unless asked.

## Custom channels and capability checks

A custom channel is a descriptor class plus a rule that turns it into Kraus
operators; register the rule on a
{py:class}`~fatqat.noise.ChannelImplementationMap` and hand the map to the
backend:

```python
import numpy as np


class BitFlip(fq.noise.Channel):
    def __init__(self, p):
        self.p = p


def bit_flip_rule(channel, *, targets):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return (
        np.sqrt(1 - channel.p) * np.eye(2, dtype=complex),
        np.sqrt(channel.p) * x,
    )


channel_map = fq.noise.default_channel_implementation_map()
channel_map.register(BitFlip, bit_flip_rule)

noise = fq.NoiseModel()
noise.add_channel(BitFlip(p=0.05), operation=op.X)
backend = fq.simulator.Simulator(
    method="DM", noise=noise, channel_implementation_map=channel_map
)
```

Rules receive the descriptor plus the gate's `targets` (so
`targets[0].register.dim` gives the dimension) and return a bare tuple of
Kraus arrays. Shapes are validated at lowering; trace preservation is not
enforced at run time — the same posture as gate matrices, which are never
unitarity-checked.

{py:meth}`~fatqat.simulator.Simulator.validate_noise` reports, without
running anything, which parts of a model a backend can execute — unknown
descriptor types come back as rejected sources.
