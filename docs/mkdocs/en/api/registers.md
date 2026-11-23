---
title: "Registers"
---

# Registers

A register groups the slots a [`Program`][fatqat.Program] can address.
Quantum slots are where operations act. Classical slots store measurement
results and values that conditions can read. Use the same register object
you passed to the program when selecting a slot: names are labels, and two
registers with the same name are still distinct.

## Register types

| Type | When to use it |
| --- | --- |
| [`QuantumRegister`][fatqat.QuantumRegister] | Use a fixed number of qubits, or specify a larger `dim` for qudits. |
| [`ClassicalRegister`][fatqat.ClassicalRegister] | Store measurement results in bits or in digits with a larger `dim`. |
| [`GridRegister`][fatqat.GridRegister] | Organize quantum slots in rows and columns so you can select parts of a grid. |
| [`ModeRegister`][fatqat.ModeRegister] | Represent optical modes in an [`LOSimulator`](simulators/linear-optics.md) program. |
| [`CountRegister`][fatqat.CountRegister] | Store photon counts measured from a `ModeRegister`. |

[`Register`][fatqat.Register] is the shared base class. Pass a concrete
quantum or classical register to `Program`.

## Register selections

Index a register, for example `q[0]`, to get a
[`RegisterRef`][fatqat.RegisterRef] for one slot. Use
[`all()`][fatqat.QuantumRegister.all] or a grid selection helper to get a
[`RegisterView`](#fatqat.RegisterView) for a group of quantum slots. See
[`Program.add()`][fatqat.Program.add] for which operations accept views.

A grid organizes logical slots. It does not place qubits on a device. Flat
indices run across each row, then continue on the next. For a
`GridRegister(2, 3)`, the helpers select these slots:

| Expression | Selected indices, in order |
| --- | --- |
| `grid.all()` | `(0, 1, 2, 3, 4, 5)` |
| `grid.row(1)` | `(3, 4, 5)` |
| `grid.column(1)` | `(1, 4)` |
| `grid.block((0, 2), (1, 3))` | `(1, 2, 4, 5)` |

## API reference { #reference }

### Register classes

::: fatqat.QuantumRegister
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.ClassicalRegister
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.GridRegister
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.ModeRegister
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.CountRegister
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.Register
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"
        - "^(?:__getitem__)$"

### References and views

::: fatqat.RegisterRef
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

<a id="fatqat.registers.RegisterView"></a>
#### class `fatqat.RegisterView` { #fatqat.RegisterView }

Immutable, hashable target returned by quantum-register selection helpers. Its
[`register`](#fatqat.RegisterView.register) attribute identifies the selected
quantum register. Obtain views from register helpers; direct construction is
unsupported.

<a id="fatqat.registers.RegisterView.register"></a>
##### attribute `register` { #fatqat.RegisterView.register }

**Type:** `fatqat.QuantumRegister`

Quantum register containing the selected members.

### Resource layouts

Use [`ResourceLayout`][fatqat.ResourceLayout] to map individual quantum slots
to device labels when the backend's default placement is not suitable.

::: fatqat.ResourceLayout
    options:
      heading_level: 4
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"

#### type `fatqat.DeviceOperand` { #fatqat.DeviceOperand }

Backend-defined opaque, hashable label for one device resource.
