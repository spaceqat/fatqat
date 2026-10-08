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
values may be complex.

## Interpolation


Built-in pulse emulators use not-a-knot spline interpolation. Two samples
give a linear curve, three give a quadratic curve, and four or more give a
cubic curve.

`Atom2LevelEmulator` uses zero outside the sample interval.
`TransmonEmulator` holds the nearest endpoint within the enclosing pulse
block, so use zero first and last samples when a control should be off outside
its grid. Every control is zero outside its block, allowing sequential pulse
operations to compose without one block contributing during another. These
rules do not change the operation's duration.

A spline can exceed the supplied sample values between points. If a model has
amplitude limits, the emulator checks the interpolated curve as well as the
samples.

## Reference


::: fatqat.emulator.SampledWaveform
    options:
      members:
        - "duration"
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
