---
title: "Measurement, reset, and barriers"
---

# Measurement, reset, and barriers

<a id="measurement-and-structural-operations"></a>
<a id="measurement-and-reset"></a>

## Measurement

Use [`measure()`][fatqat.Program.measure] or
[`measure_all()`][fatqat.Program.measure_all] to read computational-basis
outcomes into classical registers. A grouped measurement pairs targets and
outputs in the order supplied. If several results go to the same classical
destination, the last write wins.

::: fatqat.operations.Measurement
    options:
      heading_level: 3
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

## Reset

Add [`Reset`][fatqat.operations.Reset] with
[`Program.add()`][fatqat.Program.add] to return targets to `|0>` without
recording a measurement.

::: fatqat.operations.Reset
    options:
      heading_level: 3
      show_attribute_values: false

::: fatqat.operations.reset.ResetGate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

## Compiler barrier

Use [`Barrier`][fatqat.operations.Barrier] to mark a compiler boundary
without changing the simulated state.

::: fatqat.operations.Barrier
    options:
      heading_level: 3
      show_attribute_values: false

::: fatqat.operations.barrier.BarrierGate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
