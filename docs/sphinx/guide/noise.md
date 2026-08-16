# Noise

Noise is authored as physical declarations in a {py:class}`~fatqat.NoiseModel`
and passed to a backend at construction. It is separate from the program, so
the same program can run ideally or with different device models.

```python
import numpy as np
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2, 2)
program.add(op.H, 0)
program.add(op.CX, (0, 1))
program.measure((0, 1), (0, 1))

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=op.CX)
noise.add(
    fq.noise.ReadoutConfusion(
        np.array([[0.98, 0.04], [0.02, 0.96]])
    )
)

backend = fq.simulator.Simulator(method="DM", noise=noise)
result = backend.run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()
print(result.get_counts())
```

{py:meth}`~fatqat.NoiseModel.add` is the single authoring entry point. The
declaration says *what physical source exists*; `operation`, `targets`, and
`target_positions` say *where it is active*. The backend decides whether it
can realize that declaration as a finite channel, a Lindblad generator,
carrier loss, or classical measurement confusion.

## Built-in declarations

- {py:class}`~fatqat.noise.Depolarizing` `(p)` is a finite joint channel.
- {py:class}`~fatqat.noise.PauliChannel` `(terms)` is a finite qubit channel.
  The Pauli-string width sets its arity, and the string order follows the
  ordered occurrence targets (`string[0]` describes the first target).
- {py:class}`~fatqat.noise.AmplitudeDamping` accepts exactly one of finite
  probability `p` or generator `rate`, with one value per adjacent-level
  transition.
- {py:class}`~fatqat.noise.PhaseDamping` accepts exactly one of finite `p`,
  generator `rate`, or pure-dephasing time `t_phi` (`rate = 1 / t_phi`).
- {py:class}`~fatqat.noise.ThermalRelaxation` `(t1, t2)` describes total
  transverse coherence with `t2 <= 2*t1` and derives the residual pure
  dephasing rate without double-counting T1 relaxation.
- {py:class}`~fatqat.noise.Loss` `(p)` removes physical carriers on
  occupancy-aware backends.
- {py:class}`~fatqat.noise.ReadoutConfusion` stores a classical
  column-stochastic report matrix.

`PauliChannel` uses the ordinary finite-channel path; it is not a separate
registration category:

```python
noise.add(
    fq.noise.PauliChannel({"X": 0.008, "Z": 0.012}),
    operation=op.RZ,
)
noise.add(
    fq.noise.PauliChannel({"XI": 0.006, "ZZ": 0.004}),
    operation=op.CX,
)
```

## Activation scope

The presence of `operation` has one structural meaning, independent of the
descriptor parameters:

- `operation=op.X` means occurrence-bound noise. It is considered only when
  an `X` occurs.
- omitting `operation` means local background noise. It requires exactly one
  target and is meaningful only on a backend with a physical timeline.
- `targets` restricts operands. It never changes when noise is active and is
  not shorthand for “every gate.”

Matrix simulators have operation occurrences but no physical timeline. Their
finite-channel forms are:

```python
# Every X occurrence.
wide_noise = fq.NoiseModel()
wide_noise.add(fq.noise.PhaseDamping(p=0.01), operation=op.X)

# Only the exact ordered X occurrence on q0.
targeted_noise = fq.NoiseModel()
targeted_noise.add(
    fq.noise.PhaseDamping(p=0.02),
    operation=op.X,
    targets=program.quantum_registers[0][0],
)
```

Pulse emulators support operation windows and continuous background evolution:

```python
# Active only during matching X blocks.
pulse_noise = fq.NoiseModel()
pulse_noise.add(fq.noise.PhaseDamping(rate=0.002), operation=op.X)

# Active throughout elapsed pulse time on physical operand q0.
pulse_noise.add(fq.noise.PhaseDamping(t_phi=500.0), targets="q0")
```

`Barrier`, `LoadAtoms`, direct `PulseOperation`, and `Reset` have no attached
noise boundary and are rejected by `NoiseModel.add`. `Refill` is the narrow
exception on an occupancy backend: it accepts only {py:class}`~fatqat.noise.Loss`,
which is applied after refill and therefore models loading failure or immediate
post-load loss.

## Target selectors and target positions

Occurrence `targets` are an exact ordered selector. They may be all program
{py:class}`~fatqat.RegisterRef` values or all physical device labels. A scalar
is shorthand for a one-element selector. For a two-target operation, `(q0,
q1)` and `(q1, q0)` are different occurrences.

Omitting `targets` makes the registration operation-wide. Use
`target_positions` to attach a local source to selected operands of a
multi-operand occurrence:

```python
matrix_noise = fq.NoiseModel()
matrix_noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=op.CZ,
    target_positions=0,
)
matrix_noise.add(
    fq.noise.AmplitudeDamping(p=0.003),
    operation=op.CZ,
    target_positions=1,
)
```

Background dynamical noise is local in the current API: its selector is one
program reference or one physical device label, and `target_positions` is not
accepted. Correlated background generators are not silently expanded into
independent local terms.

Logical and physical selectors are separate identity spaces. They may both be
authored before a layout is known. If they later resolve to the same source,
scope, and overlapping extent, the backend rejects the actual matching event;
an alias that never matches an operation occurrence, background target, or
measurement remains a valid no-op.

## Accumulation and conflicts

Different physical source types accumulate in registration order:

```python
matrix_noise = fq.NoiseModel()
matrix_noise.add(fq.noise.AmplitudeDamping(p=0.01), operation=op.H)
matrix_noise.add(fq.noise.PhaseDamping(p=0.02), operation=op.H)
```

Background and operation-specific registrations also coexist, even for the
same source type, because they describe different activation scopes. This is
the natural pulse-model spelling for baseline damping plus extra damping
during one gate family.

FATQAT does not implement replacement or “most specific wins.” Repeated or
overlapping registrations of the same concrete declaration type in the same
operation/background scope are errors. For example, an operation-wide
`PhaseDamping` and a target-specific `PhaseDamping` cannot both match the same
`X` occurrence. Enumerate exact selectors for heterogeneous calibration, or
use disjoint `target_positions` as shown above.

## Finite channels versus generators

FATQAT performs no implicit conversion between finite probability and
continuous rate:

- matrix simulators accept backend-supported occurrence-bound finite channels
  and reject `rate`, `t_phi`, `t1`, and `t2` generator/time forms;
- pulse emulators accept backend-supported local generator/time forms and
  reject built-in finite `p` forms, even when an operation has a duration.

This keeps authored physics explicit. Use descriptor utilities when *you* own
the reference duration:

```python
relaxation = fq.noise.ThermalRelaxation(t1=60e-6, t2=80e-6)
damping, dephasing = relaxation.as_channels(duration=2e-6)

matrix_noise = fq.NoiseModel()
matrix_noise.add(damping, operation=op.H)
matrix_noise.add(dephasing, operation=op.H)
```

For pulse simulation, register the timescale declaration directly on one
target or one operation window:

```python
pulse_noise = fq.NoiseModel()
pulse_noise.add(relaxation, targets="q0")
pulse_noise.add(relaxation, operation=op.X, targets="q1")
```

The backend uses the authored generator unchanged while the pulse-block
duration determines how long it evolves. A conditionally disabled block also
disables its block-local noise; background noise remains active over elapsed
scheduled time.

## Readout confusion

{py:class}`~fatqat.noise.ReadoutConfusion` is intrinsically
measurement-bound, so `operation` and `target_positions` are forbidden:

```python
confusion = fq.noise.ReadoutConfusion(
    np.array([[0.98, 0.05], [0.02, 0.95]])
)
readout_noise = fq.NoiseModel()
readout_noise.add(confusion)                 # every measured subsystem
# or: readout_noise.add(confusion, targets="q0")
```

The matrix convention is `C[reported, true] = P(reported | true)`. Physical
collapse keeps the true outcome; only the reported classical digit is
resampled. Feedforward therefore sees the reported digit, while a reused
quantum subsystem evolves from the true post-measurement state.

Universal and targeted confusion registrations cannot coexist. Repeated
registration for the same operand is also an error. Correlated multi-operand
readout is not supported; `targets` is scalar.

## Carrier loss

{py:class}`~fatqat.noise.Loss` is not amplitude damping. On an
occupancy-aware backend, `p` is sampled independently for each selected,
currently present carrier after the matched occurrence. A hit removes the
carrier and its correlations; absence persists until `Refill`. Later
operations requiring an absent carrier are skipped, and measurement reports
the backend's absence/erasure outcome.

```python
noise.add(fq.noise.Loss(p=0.001), operation=op.RX)
```

Backends without occupancy/removal semantics reject `Loss` rather than
approximating it as a quantum channel.

## Backend lifecycle and capability checks

A backend defensively captures a noise model when constructed:

```python
noise = fq.NoiseModel()
noise.add(fq.noise.PhaseDamping(p=0.01), operation=op.X)
backend = fq.simulator.Simulator(method="DM", noise=noise)

# This affects a future backend, not `backend`.
noise.add(fq.noise.Depolarizing(p=0.02), operation=op.H)
```

Built-in declarations are immutable. Treat custom declarations as immutable
after registration. Construction rejects a captured model containing an
unsupported source. To inspect another model without constructing a new
backend, {py:meth}`~fatqat.simulator.Simulator.check_noise_support` and the
corresponding emulator methods return an advisory capability report. Program
selectors and physical labels are validated only when a concrete program and
layout are prepared.

## Execution method

For matrix simulation, `method="DM"` applies the exact Kraus sum. A
statevector cannot retain a mixed ensemble, so `method="SV"` samples a Kraus
branch per noisy occurrence and replays stochastic shots. Both implement the
same channel and converge to the same distribution. Density matrices are
usually preferable while their quadratic memory cost is affordable.

## Custom implementations

A custom finite channel subclasses {py:class}`~fatqat.noise.Channel` and has
an exact-type rule in {py:class}`~fatqat.noise.ChannelImplementationMap`:

```python
class BitFlip(fq.noise.Channel):
    _num_subsystems = 1

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
noise.add(BitFlip(p=0.05), operation=op.X)
backend = fq.simulator.Simulator(
    method="DM",
    noise=noise,
    channel_implementation_map=channel_map,
)
```

A custom pulse generator uses the separate
{py:class}`~fatqat.noise.LindbladImplementationMap`. Its rule receives
`(declaration, *, physical_dimension)` and returns local square collapse
operators. Custom physical fields are interpreted by that rule; FATQAT does
not require every generator descriptor to expose a generic `rate` property.
