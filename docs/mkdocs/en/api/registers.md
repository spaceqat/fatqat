---
title: "Registers"
---

# Registers


Registers give quantum and classical program slots stable identities. With
`Program(quantum_count, classical_count)`, each positive count creates one
default register named `"q"` or `"c"`; zero creates no register of that
kind. Construct registers explicitly when you need other names, multiple
registers, grids, metadata, or local dimensions greater than two.

## Register types


**Register choices**

| Type | Program role | Size |
| --- | --- | --- |
| [`Register`][fatqat.Register] | Common base class; not itself accepted as a quantum or classical register by [`Program`][fatqat.Program] | Explicit positive `size` |
| [`QuantumRegister`][fatqat.QuantumRegister] | Quantum operation and measurement targets | Explicit positive `size` |
| [`ClassicalRegister`][fatqat.ClassicalRegister] | Measurement outputs and condition values | Explicit positive `size` |
| [`GridRegister`][fatqat.GridRegister] | Quantum targets with rectangular selection helpers | Derived as `rows * cols` |

Names are labels and need not be unique. Keep and index the same register
objects that you pass to [`Program`][fatqat.Program]; a newly constructed register
with the same fields is not interchangeable. `metadata` is a mutable,
string-keyed mapping for application data.

Indexing with `register[index]` creates an immutable
[`RegisterRef`][fatqat.RegisterRef]. In a program with multiple registers, pass the
explicit ref rather than an ambiguous integer:

```python
import fatqat as fq
import fatqat.operations as ops

left = fq.QuantumRegister(2, name="left")
right = fq.QuantumRegister(2, name="right")
program = fq.Program([left, right])
program.add(ops.H, right[0])
```

`dim=2` creates qubits or classical bits. A larger value creates qudits or
d-ary classical digits; the quantum and classical dimensions of each
measurement pair must match. Register construction accepts every integer
dimension of at least two, but individual operations and backends may support
only some dimensions. See [Write quantum computations with Program](../guide/program.md) for a mixed qubit-qutrit
example.

::: fatqat.Register
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "^(?:__getitem__)$"

::: fatqat.QuantumRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.ClassicalRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.RegisterRef
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

## Grid selections


A [`GridRegister`][fatqat.GridRegister] arranges logical targets in row-major order; it
does not assign physical coordinates. The flat index of `(row, col)` is
`row * cols + col`, and its selection helpers return
[`RegisterView`](#fatqat.RegisterView) objects rather than tuples of refs.

For a `GridRegister(2, 3)`, the helpers select these flat indices:

**Grid selection order**

| Expression | Selected indices, in order |
| --- | --- |
| `grid.all()` | `(0, 1, 2, 3, 4, 5)` |
| `grid.row(1)` | `(3, 4, 5)` |
| `grid.column(1)` | `(1, 4)` |
| `grid.block((0, 2), (1, 3))` | `(1, 2, 4, 5)` |

Pass views to [`add`][fatqat.Program.add]. Every built-in unitary gate accepts
them. Unary gates act independently on each selected member; multi-target
gates zip corresponding members from one view per operand. All views must use
the same kind of grid selection and cardinality, and selections on the same
grid cannot overlap. Measurements and QASM export require scalar targets. The
backend validates physical placement and connectivity. See
[Write quantum computations with Program](../guide/program.md) for the ordinary Program workflow and
[Test a Program against a hardware profile](../guide/hardware-profile-simulation.md) for physical placement.

::: fatqat.GridRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

<a id="fatqat.registers.RegisterView"></a>
### class `fatqat.RegisterView` { #fatqat.RegisterView }

Immutable, hashable target returned by the grid selection helpers. Its
[`register`](#fatqat.RegisterView.register) attribute identifies the selected
grid. Obtain views from the grid helpers; direct construction is
unsupported.

<a id="fatqat.registers.RegisterView.register"></a>
#### attribute `register` { #fatqat.RegisterView.register }

**Type:** `fatqat.GridRegister`

Grid register containing the selected members.


## Resource layouts


A [`ResourceLayout`][fatqat.ResourceLayout] associates scalar quantum
[`RegisterRef`][fatqat.RegisterRef] operands with device labels. Most applications can
use the backend's default layout; pass `resource_layout=` when you need a
specific supported placement. Each backend defines the labels it accepts and
checks coverage, uniqueness, dimensions, placement, and connectivity when the
program runs. [`PulseOperation`][fatqat.operations.PulseOperation] channels address the
emulator model directly and do not use this layout.

::: fatqat.ResourceLayout
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

### type `fatqat.DeviceOperand` { #fatqat.DeviceOperand }

Backend-defined opaque, hashable label for one device resource.
