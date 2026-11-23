---
title: "Atom-array operations"
---

# Atom-array operations

Use these operations with
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] to load atoms
and change which pairs can interact. Other built-in backends do not implement
them.

**Atom-array operations**

| Value | Targets | Effect | Conditions | Attached noise |
| --- | --- | --- | --- | --- |
| [`Put`][fatqat.operations.Put] | One or more scalars, or one [`RegisterView`](../registers.md#fatqat.RegisterView) | Loads `\|0>` into each empty site; leaves occupied sites unchanged. | Allowed. | [`Loss`][fatqat.noise.Loss] only, after each enabled `Put` operation. |
| [`Pair`][fatqat.operations.Pair] | Exactly two scalars | Adds their undirected connectivity edge; repeated pairing is a no-op. | Rejected. | [`Loss`][fatqat.noise.Loss] or a supported finite channel. |
| [`Unpair`][fatqat.operations.Unpair] | Exactly two scalars | Removes their edge; removing an absent edge is a no-op. | Rejected. | [`Loss`][fatqat.noise.Loss] or a supported finite channel. |

Sites start empty by default. An empty site does not bypass gate or pairing
checks: an unsupported gate or unpaired `CZ` still fails validation.
See [AtomArraySimulator](../simulators/atom-array.md) for occupancy,
measurement and noise behavior across a run.

## API reference { #reference }

::: fatqat.operations.Put
    options:
      heading_level: 3
      show_attribute_values: false

::: fatqat.operations.put.PutGate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.operations.Pair
    options:
      heading_level: 3
      show_attribute_values: false

::: fatqat.operations.pairing.PairGate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.operations.Unpair
    options:
      heading_level: 3
      show_attribute_values: false

::: fatqat.operations.pairing.UnpairGate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
