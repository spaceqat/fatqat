---
title: "Noise model"
---

# Noise model


[`NoiseModel`][fatqat.NoiseModel] collects noise rules and says where each one
applies: on matching operations, throughout pulse time, or when reporting
measurements. A model can combine simulator channels, emulator Lindblad
operators, carrier loss, and readout confusion. Each backend accepts only the
rules it can implement.

## Build a model


This example adds phase damping after every `RX`, amplitude damping to the
second operand of every `CZ`, and readout confusion on `q[0]`. Logical targets
use references from the program; device-label targets use physical labels
defined by the backend.

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
q = program.quantum_registers[0]
program.add(ops.RX(0.4), q[0])
program.add(ops.CZ, (q[0], q[1]))
program.measure_all()

noise = fq.NoiseModel()
noise.add(
    fq.noise.PhaseDamping(p=0.01),
    operation=ops.RX,
)
noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=ops.CZ,
    target_positions=1,
)
noise.add(
    fq.noise.ReadoutConfusion(
        [[0.98, 0.04], [0.02, 0.96]]
    ),
    targets=q[0],
)

simulator = fq.simulator.Simulator(method="DM", noise=noise)
result = simulator.run(
    program,
    shots=1_000,
    simulation_config={"seed": 7},
    result_config={"counts": True},
).result()
```

## Where noise applies


Pass `operation` for noise tied to a matching operation. A simulator applies
a channel or samples loss after the operation; an emulator keeps the matching
Lindblad operators active during the pulse. Omit `operation` for background
noise; this does not mean “on every gate.” Readout confusion always applies at
measurement.

**Noise scopes**

| Scope | Noise type | `operation` | `targets` | `target_positions` |
| --- | --- | --- | --- | --- |
| On matching operations | [`Channel`][fatqat.noise.Channel] or [`Loss`][fatqat.noise.Loss] | Required operation instance or subclass | Optional exact ordered target selector | Optional affected positions in the operation's target order |
| During pulse time (background) | A [`Channel`][fatqat.noise.Channel] that acts on one subsystem | Omitted or `None` | Exactly one logical reference or device label | Must be omitted or `None` |
| At measurement (readout) | [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | Must be omitted | Omitted for universal readout, or one scalar selector | Must be omitted |

For readout confusion, both `operation` and `target_positions` must be
absent. Passing either keyword explicitly, even as `None`, is an error.

Background noise uses local Lindblad operators over elapsed emulator time.
[`Loss`][fatqat.noise.Loss], probability-form
[`Depolarizing`][fatqat.noise.Depolarizing], and other noise types that do not act on
exactly one subsystem cannot be used this way. Background noise remains active
regardless of an individual operation's classical condition.

`Barrier`, `Reset`, and direct
[`PulseOperation`][fatqat.operations.PulseOperation] controls have no attachable noise
boundary. `Put` accepts only [`Loss`][fatqat.noise.Loss]; the loss is applied
after loading and models loading inefficiency. See
[Backend support](backend-support.md#noise-backend-support) for the scopes and noise forms implemented
by each built-in backend.

## Match operations


You may pass either an operation value or an operation class. For example,
`operation=ops.X` matches the exported `X` singleton, while
`operation=ops.RX` matches every `RX(...)` angle. Parameters do not narrow
the match, and registering a base operation class does not include its
subclasses.

Noise attached to an operation follows its classical condition. If the
condition is false, neither the operation nor its noise runs. A simulator
applies compatible channels after the operation, in registration order. An
emulator keeps compatible Lindblad operators active together during the
matching pulse.

## Match targets


`targets` may use logical [`RegisterRef`][fatqat.RegisterRef] values from the
program or hashable device labels accepted by the backend. The run's
[`ResourceLayout`][fatqat.ResourceLayout] maps program references into that
device-label space.

The legal device-label set is backend-specific:

| Built-in backend | Legal device labels |
| --- | --- |
| Generic [`Simulator`][fatqat.simulator.Simulator] | Labels in the effective resource layout |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | Labels in the effective resource layout; the program defines the sites |
| [`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] and [`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] | Every label in the configured grid |
| Pulse emulators | Every label in the configured model or atom arrangement |

**Accepted `targets` forms**

| Form | Scope | Meaning and constraints |
| --- | --- | --- |
| `None` | Operation | Match every operation of the exact class. |
| One `RegisterRef` or non-tuple hashable label | Operation | Shorthand for a one-element selector. An operation with a known width must take one target; a variadic operation is checked later. |
| Non-empty tuple of `RegisterRef` values | Operation | Exact ordered logical target selector. Its length must equal a known operation width. |
| Non-empty tuple of hashable device labels | Operation | Exact ordered device-label selector. Its length must equal a known operation width. |
| One scalar or one-element tuple | Background | Select exactly one logical reference or device label. |
| `None` | Readout | Select every measured operand. Universal and targeted readout registrations cannot coexist. |
| One `RegisterRef` or non-tuple hashable label | Readout | Select one measured logical reference or device label. Correlated readout is not supported. |

Lists and [`RegisterView`](../registers.md#fatqat.RegisterView) values are not accepted. A target
tuple cannot mix logical references and device labels. FATQAT always treats
a tuple as the complete ordered selector, so nest tuple-valued device
labels: `targets=(("site", 0),)` selects one such label, and
`targets=(("site", 0), ("site", 1))` selects an ordered pair. Readout noise
cannot target a tuple-valued device label.

The selector order must match the operation's target order in the program. A
noise rule targeting `(q[0], q[1])` does not match the same operation on
`(q[1], q[0])`. Logical and device-label selectors use the same ordering
convention.

## Select affected operands


After an operation matches, `target_positions` chooses which operands
receive the noise. Pass one integer or a nonempty, strictly increasing tuple
of nonnegative integers. Positions use zero-based operation order; `None`
selects every operand.

```python
local_noise = fq.NoiseModel()
local_noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=ops.CZ,
    target_positions=0,
)
local_noise.add(
    fq.noise.AmplitudeDamping(p=0.003),
    operation=ops.CZ,
    target_positions=1,
)
```

The number of selected operands must match the number of subsystems the noise
type acts on. [`add`][fatqat.NoiseModel.add] checks this immediately when both
sizes are known; variadic operations are checked when the program runs.

## Combine noise sources


[`add`][fatqat.NoiseModel.add] appends; it never replaces a registration.
Different noise types may act on the same operation. The same type may also be
used on disjoint targets or operand positions. Background and operation noise
are independent, so the same type can appear in both scopes.

FATQAT rejects overlapping rules of the same type for the same operation or
background scope. A rule for every matching operation overlaps an exact target
selector, and selecting every operand overlaps a positional selection.

Logical and device-label selectors cannot be compared until a resource layout
is available. They may therefore both be added, but execution raises
[`BackendValidationError`][fatqat.errors.BackendValidationError] if the layout makes both
select the same noise type and operands for an actual operation.

Readout is unique per measured operand. Duplicate universal or duplicate exact
selectors fail during `add`; universal and targeted registrations cannot be
mixed. A logical selector and a device-label selector that name the same
measured operand are rejected when the program runs.

## When validation happens


**Validation stages**

| When | Checks | Typical failure |
| --- | --- | --- |
| Creating a noise value | Probabilities, rates, matrices, finite values, and type-specific parameter relationships | `TypeError` or `ValueError` from the noise type |
| [`add`][fatqat.NoiseModel.add] | Noise type, operation boundary, target form, known width, position ordering/range, and immediately visible conflicts | `TypeError` or `ValueError`; the failed addition changes nothing |
| Creating the backend | Whether the configured backend accepts every noise form and scope | [`BackendValidationError`][fatqat.errors.BackendValidationError] |
| Running a program | Whether references belong to the program, device labels belong to the backend's legal device universe, selectors resolve without conflicts, and dimensions and the chosen execution method are compatible | Usually [`BackendValidationError`][fatqat.errors.BackendValidationError] or [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] |

A legal selector can still match nothing in a run; in that case, it is a no-op.
Some checks require a concrete program, resource layout, and execution method.

## Validate backend support


Use [`validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] for a simulator or
[`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] for a pulse
emulator to validate a candidate model. The method returns `None` when the
configured backend accepts the model. Otherwise, it raises
[`BackendValidationError`][fatqat.errors.BackendValidationError] whose message includes every
model-level rejection reason.

```python
probe = fq.simulator.Simulator(method="DM")
probe.validate_noise_model(noise)
```

Program- and layout-dependent checks still happen when the model is used.

## API


::: fatqat.NoiseModel
    options:
      members:
        - "add"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
