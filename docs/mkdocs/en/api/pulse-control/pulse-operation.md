---
title: "PulseOperation"
---

# PulseOperation

!!! warning "Pulse control under development"

    The pulse control module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Add a [`PulseOperation`][fatqat.operations.PulseOperation] with
`program.add(operation)`. Its controls already specify the physical channels
to drive. Put controls in the same block when they should share a time axis.
Each control's `start_offset` is measured from the start of that block.

## Conditions and noise


`TransmonEmulator` allows `program.add(operation, condition=...)`. If the
condition is false, the controls are skipped but the block still takes its full
duration. Model drift and background Lindblad noise continue during that time.
`Atom2LevelEmulator` does not support conditions.

Operation-scoped noise cannot be attached to a direct pulse block, so
`noise.add(..., operation=ops.PulseOperation)` raises `ValueError`.
Background noise selected by target or device label still applies.

## Drawing and export { #support }


Use supported gate-level operations for circuit drawing and OpenQASM export.

## API reference { #reference }


::: fatqat.operations.PulseOperation
    options:
      heading_level: 3
      members: false
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
