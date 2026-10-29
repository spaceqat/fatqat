---
title: "Gate realization"
---

# Gate realization

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

[`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] maps ordinary gates to pulse
definitions. A direct [PulseOperation](pulse-operation.md) already contains its controls and
does not use this map.

## Rules


A rule is called as `rule(operation, *, device_operands=...)` and must return
a [`PulseDefinition`][fatqat.emulator.PulseDefinition]. Register a general rule to handle every ordered
device-operand tuple, or register separate rules for specific tuples. A
tuple-specific entry may also be a fixed definition or a callable that only
accepts `operation`. The tuple contains ordered physical labels such as
`("q0", "q1")`, not program register references.

Both modes accept `num_subsystems=None`; device-specific tuples may then have
different lengths. Otherwise, each tuple must match the declared target count.

## Definitions


A [`PulseDefinition`][fatqat.emulator.PulseDefinition] contains a duration, a tuple of controls, and
optional `PhaseShift` or `PhaseSwap` actions. Conditions and noise remain
on the operation in the program.

`PhaseShift` changes one model frame after the pulse. `PhaseSwap` exchanges
two frames. Direct pulse operations do not have post-actions.

The emulator calls and validates a selected rule when the gate is used. Raise
[`BackendValidationError`][fatqat.errors.BackendValidationError] to report unsupported operands
or parameters. Other exceptions and return values that are not
`PulseDefinition` are reported as `PulseImplementationError`.

The [transmon](../pulse-emulator.md) and
[neutral-atom](../atom-emulators.md) pages show the built-in maps and full
workflows.

## Reference


::: fatqat.emulator.PulseImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PulseDefinition
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseShift
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseSwap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
