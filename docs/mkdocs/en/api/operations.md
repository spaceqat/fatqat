---
title: "Operations"
---

# Operations


Import `fatqat.operations` as `ops`, then add operations to a program with
[`fatqat.Program.add`][fatqat.Program.add]:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.H, 0)            # ready-to-use operation
program.add(ops.RX(0.2), 1)      # parameterized operation
program.add(ops.CX, (0, 1))      # ordered targets
```

Parameter-free operations such as `ops.H` and `ops.Reset` are ready to use
without parentheses. Construct parameterized gates and
[`PulseOperation`][fatqat.operations.PulseOperation] values before adding them. Create measurements with
[`measure`][fatqat.Program.measure] or [`measure_all`][fatqat.Program.measure_all].

## Reference pages


**Operation families**

| Page | Contents |
| --- | --- |
| [Qubit gates](operations/qubit-gates.md) | Fixed and parameterized qubit gates, exact target order, matrices, and constructor reference. |
| [Qudit gates](operations/qudit-gates.md) | Qudit gates, level constraints, and basis actions. |
| [Measurement and structural operations](operations/structural.md) | Measurement and reset behavior, and compiler barriers. |
| [Atom-array operations](operations/atom-gates.md) | Atom-array occupancy, pairing, and attached-noise constraints. |
| [PulseOperation](pulse-control/pulse-operation.md) | Channel-addressed `PulseOperation`—still imported from `fatqat.operations`—with its timing and backend support. |

- [Qubit gates](operations/qubit-gates.md)
- [Qudit gates](operations/qudit-gates.md)
- [Measurement and structural operations](operations/structural.md)
- [Atom-array operations](operations/atom-gates.md)

## Construction


For target-based operations, [`add`][fatqat.Program.add] resolves target
references, checks arity, and rejects repeated scalar targets. The selected
backend checks operation and device support when you submit the program; an
unsupported family raises
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]. A direct
[`PulseOperation`][fatqat.operations.PulseOperation] follows the channel-addressing rules on
[PulseOperation](pulse-control/pulse-operation.md) and is added without targets.

Most targets are a scalar [`RegisterRef`][fatqat.RegisterRef] or an integer. Use an
integer when the program has one quantum register; with multiple registers,
index the register you want. Controlled gates use control-first order, and the
first local operand is the most-significant digit in the matrices and basis
actions on the family pages.

[`RX`][fatqat.operations.RX], [`RY`][fatqat.operations.RY], and [`RZ`][fatqat.operations.RZ] accept one
[`RegisterView`](registers.md#fatqat.RegisterView); [`CX`][fatqat.operations.CX] and [`CZ`][fatqat.operations.CZ] accept two
compatible views and pair their members in order. See [Registers](registers.md) for the
view compatibility rules and [Write quantum computations with Program](../guide/program.md) for the ordinary
construction workflow.

## Operation base


Subclassing [`Operation`][fatqat.operations.Operation] defines a new program-level value; it does
not register a matrix or pulse realization. See [Matrix implementations](implementation.md) for the
custom matrix contract and [Gate realization](pulse-control/gate-realization.md) for pulse
realizations.

::: fatqat.operations.Operation
    options:
      members:
        - "name"
        - "num_subsystems"
        - "min_targets"
        - "accepts_views"
        - "validate_targets"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
