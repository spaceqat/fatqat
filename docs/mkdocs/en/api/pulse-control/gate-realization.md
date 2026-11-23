---
title: "Custom gate pulses"
---

# Custom gate pulses { #gate-realization }

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Use a [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] when
you want to supply your own waveform for a gate or choose a different pulse
for a particular physical target. The program still specifies gates and
logical targets. When the emulator prepares the program, it uses the gate
and its mapped physical targets to select a pulse definition. To drive a
channel directly from the program, use a [PulseOperation](pulse-operation.md).

## Define the pulse { #definitions }

Specify the duration and physical controls in a
[`PulseDefinition`][fatqat.emulator.PulseDefinition]. A definition can also
apply frame actions after the pulse interval. `PhaseShift` shifts one frame
and `PhaseSwap` exchanges two frames. Conditions and noise remain on the
original gate operation in the program.

## Register the implementation { #rules }

Use a general rule if the same callable can handle every device-operand tuple.
To supply a pulse for an ordered set of physical labels, such as `("q0", "q1")`,
register it as a device-specific rule. Pass the map to the emulator as
`gate_implementation_map=`. The [`PulseImplementationMap` example][fatqat.emulator.PulseImplementationMap]
below registers a fixed definition and runs the gate with that map. See
[`PulseImplementationMap.add()`][fatqat.emulator.PulseImplementationMap.add]
for the accepted rule forms and tuple lengths.

The emulator calls and validates the selected rule when the gate is used.
See [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] for
errors that can occur when a rule runs.

The [transmon](../pulse-emulator.md) and [neutral-atom](../atom-emulators.md)
API pages describe the default maps and constructor options for each emulator.

## API reference { #reference }


::: fatqat.emulator.PulseImplementationMap
    options:
      heading_level: 3
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PulseDefinition
    options:
      heading_level: 3
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseShift
    options:
      heading_level: 3
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseSwap
    options:
      heading_level: 3
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
