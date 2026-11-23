---
title: "Pulse control"
---

# Pulse control

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

A pulse emulator runs physical controls with specified waveforms, channels,
and timing. When the program contains a gate, the emulator uses the gate's
pulse implementation to obtain those controls. A direct pulse operation
specifies the controls in the program.

## Choose a pulse interface { #direct-pulses-and-gate-pulses }

| What you want to do | What goes in the program | Where the pulse comes from |
| --- | --- | --- |
| Run a gate with its existing pulse implementation | A gate and its logical targets, such as `program.add(ops.X, 0)` | The emulator's gate implementation map |
| Specify a waveform on a physical channel | A [PulseOperation](pulse-operation.md) | A [SampledWaveform](sampled-waveform.md) assigned to a channel through [PulseControl](pulse-control.md) |
| Supply your own pulses for a gate | A gate and its logical targets | A [`PulseDefinition`][fatqat.emulator.PulseDefinition] registered in a [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap]. See [Custom gate pulses](gate-realization.md). |

For example, suppose logical target `0` maps to physical transmon `q0`.
For `program.add(ops.X, 0)`, the emulator looks up the `X` implementation for
`q0` when it prepares the program. For a direct pulse, select
`model.control.drive("q0")` when creating the `PulseControl`, then add its
block with `program.add(pulse)`. The control already carries the physical
channel.

By default, [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] provides a
built-in gate implementation map. The default map for
[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] is empty, so ordinary
gates there require a custom map.

<a id="built-in-support"></a>

Durations, offsets, and sample times use the model's time unit. For channels,
waveform units, and limits, see [Transmon controls](../pulse-emulator.md#direct-controls)
or [two-level atom controls](../atom-emulators.md#model-and-controls).

## Validation

The value constructors check the sample grid, offsets, and block timing,
including whether controls fit within the block and use distinct channels.
Channel selectors check the form of an address, such as whether a subsystem
label is nonempty. When the emulator prepares the block for execution, it
checks the channel's model family and whether the named resources exist. It
also checks real or complex sample requirements, amplitude and duration
limits, and which channels can run together.

<a id="pulse-probability-noise"></a>


## Continuous-time noise


Pulse emulators use Lindblad noise specified by rates or relaxation times.
See the [supported pulse noise models](../noise/backend-support.md#noise-emulator-support)
for each emulator.
