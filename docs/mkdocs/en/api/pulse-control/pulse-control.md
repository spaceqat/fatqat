---
title: "PulseControl"
---

# PulseControl

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

[`PulseControl`][fatqat.emulator.PulseControl] assigns a waveform to one
physical control channel at an offset within a pulse block. Select the
channel from `model.control`, for example `model.control.drive("q0")`, then
include the control in a [PulseOperation](pulse-operation.md) or a
[`PulseDefinition`](gate-realization.md#fatqat.emulator.PulseDefinition).

See the [model control references](index.md#built-in-support) for channels
and units, and [SampledWaveform](sampled-waveform.md#interpolation) for how
emulators handle samples between and outside the grid points.

## API reference { #reference }


::: fatqat.emulator.PulseControl
    options:
      heading_level: 3
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.ControlChannel
    options:
      heading_level: 3
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
