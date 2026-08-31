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

If a program contains `Put`, every declared site starts empty for every shot.
Sites are populated only when `Put` runs, and a later `Put` can reload a
lost atom. A [`Loss`][fatqat.noise.Loss] declaration attached to `Put`
shares the operation's condition and runs after every matching `Put`
operation whose condition passes, even when the site was already occupied and
the `Put` itself did nothing.

For convenience, `program.add(ops.Put, register.all())` expands the view into
one variadic `Put` operation targeting every member of that quantum register.

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
