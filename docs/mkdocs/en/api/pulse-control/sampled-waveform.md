---
title: "SampledWaveform"
---

# SampledWaveform

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

[`SampledWaveform`][fatqat.emulator.SampledWaveform] describes a signal on a local time grid. Times use
the model's time unit, while the channel determines the sample unit and whether
values may be complex. The sample grid starts at zero, and a
[`PulseControl`][fatqat.emulator.PulseControl] places it at an offset within
the enclosing pulse block.

## Interpolation


See the [`SampledWaveform` reference][fatqat.emulator.SampledWaveform] for
the spline degree used between samples. Outside the sample interval,
emulators handle controls differently.

- `Atom2LevelEmulator` uses zero.
- `TransmonEmulator` holds the nearest endpoint within the pulse block. Set
  the first and last samples to zero if the control should be off outside the
  sample interval.

Every control is zero outside its block. The block duration is set by the
enclosing pulse operation or definition.

A spline can exceed the supplied sample values between points. If a model has
amplitude limits, the emulator checks the interpolated curve as well as the
samples.

## API reference { #reference }


::: fatqat.emulator.SampledWaveform
    options:
      heading_level: 3
      members:
        - "duration"
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
