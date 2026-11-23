---
title: "Program"
---

# Program

A [`Program`][fatqat.Program] describes a quantum computation. Its registers
identify the quantum and classical slots it uses, and its instructions record
which operations and measurements to perform and in what order. A backend
executes the instructions and checks whether it supports them.

<a id="registers"></a>

Quantum registers identify the slots that operations act on. Classical
registers hold measurement results and values that conditions can read.
See [Registers](registers.md) for register types and how to select their slots.

## Targets

Targets tell an operation which slots to act on. For example, in
`program.add(ops.CX, (0, 1))`, the two indices select the control and target
from the program's only quantum register. The same addressing rules apply
when selecting measurement outputs or the classical slots a condition reads.

| Form | Accepted by | When to use it |
| --- | --- | --- |
| Integer | `add()`, `measure()`, `measure_all(outputs=...)`, conditions | Use a zero-based index when the program has exactly one register of the required kind. |
| [`RegisterRef`][fatqat.RegisterRef] | `add()`, `measure()`, `measure_all(outputs=...)`, conditions | Use `q[0]` or `c[0]` to select a slot from a specific register. Required when there is more than one register of that kind. |
| [`RegisterView`](registers.md#fatqat.RegisterView) | `add()` with gates that support views, or `Put` | A single-target gate acts on each selected slot. A multi-target gate uses one slot from each view at each position. `Put` takes one view as its complete target collection. |

For a controlled gate, list controls before targets. For a
[`PulseOperation`][fatqat.operations.PulseOperation], omit `targets`: the pulse
channels specify the physical resources directly.

<a id="conditions"></a>
<a id="measurements"></a>
<a id="program-templates"></a>
<a id="parameter-binding"></a>
<a id="draw"></a>
<a id="logical-interaction-frequency"></a>
<a id="read-instructions"></a>

## API reference { #reference }

::: fatqat.Program
    options:
      heading_level: 3
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"

### Parameter values

::: fatqat.Parameter
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.ParameterVector
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

### Helpers

::: fatqat.program.OperationInstruction
    options:
      heading_level: 4

::: fatqat.visualization.to_qutip_circuit
    options:
      heading_level: 4
