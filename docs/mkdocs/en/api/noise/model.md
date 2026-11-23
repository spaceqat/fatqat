---
title: "Noise model"
---

# Noise model

A [`NoiseModel`][fatqat.NoiseModel] holds the noise rules for a run. Use
[`add()`][fatqat.NoiseModel.add] to specify the noise type, when it acts, and
which operands receive it. When you pass the model to a built-in backend, the
backend takes a copy of the registrations. To try new rules, pass the updated
model to a new backend.

## Build a model

The first rule applies to every `RX`. The second applies to the second operand of
every `CZ`. The third applies when measuring `q[0]`.

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
    execution_config={"seed": 7},
    result_config={"counts": True},
).result()
```

## Where noise applies

**Noise scopes**

| Scope | Noise type | `operation` | `targets` | `target_positions` |
| --- | --- | --- | --- | --- |
| On matching operations | [`Channel`][fatqat.noise.Channel] or [`Loss`][fatqat.noise.Loss] | Required operation instance or subclass | Optional exact ordered target selector | Optional affected positions in the operation's target order |
| During pulse time (background) | A [`Channel`][fatqat.noise.Channel] that acts on one subsystem | Omitted or `None` | Exactly one logical reference or device label | Must be omitted or `None` |
| At measurement (readout) | [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | Must be omitted | Omitted for universal readout, or one scalar selector | Must be omitted |

A simulator applies operation noise after a matching operation. A pulse emulator
applies operation noise during the matching pulse and background noise throughout
elapsed time. See [Backend support](backend-support.md#noise-backend-support) for
available scopes.

## Match operations

Use `operation=ops.RX` to match every `RX(...)`, regardless of angle. Passing an
operation instance also matches its exact class. When a classical condition skips
an operation, its attached noise is skipped too.

## Match targets

Use `targets` to pick which occurrences receive noise. A logical
[`RegisterRef`][fatqat.RegisterRef] follows the program operand. A device label
follows the physical location chosen by the
[`ResourceLayout`][fatqat.ResourceLayout]. An ordered pair such as `(q[0], q[1])`
matches only that operand order.

The legal device-label set is backend-specific:

| Built-in backend | Legal device labels |
| --- | --- |
| [`Simulator`][fatqat.simulator.Simulator] or [`LOSimulator`][fatqat.simulator.LOSimulator] | Labels in the effective resource layout |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | Labels in the effective resource layout; the program defines the sites |
| [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] | Every configured device label |
| Pulse emulators | Every label in the configured model or atom arrangement |

To name a tuple-valued device label, wrap it in a selector tuple, for example
`targets=(("site", 0),)`. Readout selectors take a single non-tuple label or
reference. For the full selector rules, see [`add()`][fatqat.NoiseModel.add].

## Select affected operands

Once an operation matches, `target_positions` picks the affected operands by
zero-based position. For example, `target_positions=1` applies the channel only to
the second operand, as in the `CZ` rule above. Omit it to use all operands. The
number selected must match the width of the noise type.

## Combine noise sources

[`add()`][fatqat.NoiseModel.add] appends rules. Different noise types can affect
the same operand. Rules of the same type can use disjoint operands or separate
scopes. Overlapping rules of the same type in the same scope are rejected, except
that each [`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] adds its own
jump. Only one readout rule can apply to a measured operand.

<a id="when-validation-happens"></a>

## Validate backend support

Use [`validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] for
a simulator or
[`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model]
for a pulse emulator to check a model without running a program. When executing a
program, the backend also checks resolved targets and dimensions. A valid rule
that matches nothing has no effect.

## API reference { #api }

::: fatqat.NoiseModel
    options:
      members:
        - "add"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
