---
title: "Operations"
---

# Operations

An operation describes what to apply to a program's targets. Pass
parameter-free operations such as `ops.H` directly to
[`Program.add()`][fatqat.Program.add]. Call a parameterized constructor such
as `ops.RX(0.2)` first.

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.RX(0.2), 1)
program.measure_all()
```

Create measurements with [`measure()`][fatqat.Program.measure] or
[`measure_all()`][fatqat.Program.measure_all]. For target forms and conditions,
see [`Program.add()`][fatqat.Program.add]. The backend checks whether it can
run the requested operations.

<a id="construction"></a>

## Reference pages

| Page | Use it to look up |
| --- | --- |
| [Qubit gates](operations/qubit-gates.md) | Fixed gates and rotations on qubits. |
| [Qudit gates](operations/qudit-gates.md) | Gates on multilevel systems, including level and dimension requirements. |
| [Optical gates](operations/optical-gates.md) | Phase shifts and beam splitters on optical modes. |
| [Atom-array operations](operations/atom-gates.md) | Loading atoms and changing their pairings. |
| [Measurement, reset, and barriers](operations/structural.md) | Readout, reset to the ground state, and compiler barriers. |
| [PulseOperation](pulse-control/pulse-operation.md) | Direct pulse controls, addressed by channel rather than by register target. |

## Operation base


When defining a custom [`Operation`][fatqat.operations.Operation], use
`accepts_views` to opt into grouped targets and `validate_targets()` to
check parameters against the target registers. To make the operation runnable,
see [Matrix implementation maps](implementation.md) or
[Custom gate pulses](pulse-control/gate-realization.md).

::: fatqat.operations.Operation
    options:
      heading_level: 3
      members:
        - "name"
        - "num_subsystems"
        - "min_targets"
        - "accepts_views"
        - "validate_targets"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
