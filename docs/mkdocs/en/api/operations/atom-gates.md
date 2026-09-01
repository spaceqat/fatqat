---
title: "Atom-array operations"
---

# Atom-array operations


`Put` manages atom occupancy; `Pair` and `Unpair` manage connectivity.
They are not unitary matrix gates. Add them with
[`fatqat.Program.add`][fatqat.Program.add]. [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator]
is the only built-in backend that implements them. Other matrix and pulse
backends raise [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError].

**Atom-array operations**

| Value | Targets | Effect | Conditions | Attached noise |
| --- | --- | --- | --- | --- |
| [`Put`][fatqat.operations.Put] | One or more scalars, or one [`RegisterView`](../registers.md#fatqat.RegisterView) | Loads `\|0>` into each empty site; leaves occupied sites unchanged. | Allowed. | [`Loss`][fatqat.noise.Loss] only, after each enabled `Put` operation. |
| [`Pair`][fatqat.operations.Pair] | Exactly two scalars | Adds their undirected connectivity edge; repeated pairing is a no-op. | Rejected. | [`Loss`][fatqat.noise.Loss] or a supported finite channel. |
| [`Unpair`][fatqat.operations.Unpair] | Exactly two scalars | Removes their edge; removing an absent edge is a no-op. | Rejected. | [`Loss`][fatqat.noise.Loss] or a supported finite channel. |

Every declared site starts empty on each shot. `Put` is the only operation
that loads an atom, and a later `Put` can refill a lost site. Until a site is
loaded, supported gates and reset have no effect there, and measurement
reports `2`. Native-gate and pairing checks still run first, so an empty site
cannot conceal an unsupported gate or unpaired `CZ`.

A [`Loss`][fatqat.noise.Loss] declaration attached to `Put` shares the
operation's condition and runs after every matching `Put` whose condition
passes, even if the target was already occupied and loading did nothing.

To load an entire named register, use
`program.add(ops.Put, register.all())`; the view expands into one variadic
`Put` over its members. For a program created from an integer site count, use
`program.add(ops.Put, tuple(range(num_atoms)))`.

`Pair` and `Unpair` update the connectivity used by later supported gates;
they do not change the quantum state or make an unsupported gate available. In
the built-in atom-array profile, [`CZ`][fatqat.operations.CZ] is native and requires a current
pairing. The atom backend rejects a condition on either instruction with
[`BackendValidationError`][fatqat.errors.BackendValidationError] when the program runs.

::: fatqat.operations.Put
    options:
      show_attribute_values: false

::: fatqat.operations.Pair
    options:
      show_attribute_values: false

::: fatqat.operations.Unpair
    options:
      show_attribute_values: false
