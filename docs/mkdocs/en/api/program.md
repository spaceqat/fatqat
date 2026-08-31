---
title: "Program"
---

# Program


[`Program`][fatqat.Program] records a quantum workload without tying it to a
device. Add operations and measurements in execution order, then choose a
backend when you run it.

FATQAT catches malformed targets, measurements, and conditions while you build
the program. The backend checks whether it supports the requested operations,
dimensions, placement, and feedforward.

```python
import fatqat as fq
import fatqat.operations as ops

bell = fq.Program(2, 2, metadata={"name": "bell"})
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))
bell.measure_all()
```

## Registers


The two register arguments accept these forms:

**Register inputs**

| Form | Result | Constraints |
| --- | --- | --- |
| Non-negative integer `n` | `n > 0` creates one dimension-2 register named `"q"` or `"c"`; `0` creates no register. | The value must be an exact integer, not a boolean. A negative value is rejected. |
| List or tuple of registers | Preserves the supplied register objects in a tuple. | Every item must have the matching quantum or classical register kind. Use this form for names, multiple registers, grids, or qudits. |

See [Registers](registers.md) for explicit registers, dimensions, and grid selections.

## Targets


A bare integer indexes the sole register of the relevant kind; it is not a
global index across several registers. With multiple registers, index the
register you want and pass the resulting [`RegisterRef`][fatqat.RegisterRef].

**Target forms**

| Form | Accepted by | Rule |
| --- | --- | --- |
| Integer | [`add`][fatqat.Program.add], [`measure`][fatqat.Program.measure], and conditions | Requires exactly one register of the relevant kind and must be within its zero-based bounds. |
| [`RegisterRef`][fatqat.RegisterRef] | [`add`][fatqat.Program.add], [`measure`][fatqat.Program.measure], and conditions | Must have the required register kind and come from a register in this program. |
| [`RegisterView`](registers.md#fatqat.RegisterView) | [`add`][fatqat.Program.add] only | Every built-in unitary gate accepts views. Unary gates map over one view; multi-target gates zip one compatible view per operand. `Put` accepts one view as its complete target collection. Measurement does not accept views. |

A [`PulseOperation`][fatqat.operations.PulseOperation] does not use the target forms
above. Add it with `program.add(operation)` and no `targets` argument.
See [PulseOperation](pulse-control/pulse-operation.md) for details.

For other operations, `targets` is one tuple in operand order. Controlled
gates list controls before targets. See [Registers](registers.md) for view selection and
compatibility, and [Write quantum computations with Program](../guide/program.md) for the ordinary construction
workflow.

## Conditions


Pass `condition=(slot, literal)` to [`add`][fatqat.Program.add], or pass a
non-empty tuple or list of such pairs for logical AND. A slot follows the same
integer-or-ref rules as other classical operands. Each literal is a Python
integer in `0 <= literal < slot.dim`; booleans are also accepted. Every term
is compared with the current classical value when the operation is reached.

FATQAT checks each condition when the operation is added. A condition may refer
to an unmeasured slot, whose initial value is zero; an earlier measurement
replaces that value. The backend decides whether it supports feedforward.

## Measurements


[`measure`][fatqat.Program.measure] pairs quantum targets with classical
outputs positionally. Both sides must be non-empty, have the same length, and
have matching dimensions at every position. Repeated operands are processed
in pair order; see [Measurement and structural operations](operations/structural.md) for measurement behavior.

[`measure_all`][fatqat.Program.measure_all] flattens all registers and their members
in declaration order and appends one grouped measurement. It requires equal,
non-zero quantum and classical counts and matching dimensions at every
position. Read [Write quantum computations with Program](../guide/program.md) for a mid-program measurement and
feedforward workflow.

<a id="program-templates"></a>


## Parameter binding


Parameters are immutable identity objects. Names are labels only: two
`Parameter("theta")` objects are different binding keys. Reuse one object
when several operation arguments should share a value.

**Binding forms**

| Mapping key | Accepted value | Constraint |
| --- | --- | --- |
| [`Parameter`][fatqat.Parameter] | Built-in integer or float, or NumPy integer or floating scalar | The same object must be supplied directly to an operation parameter. |
| [`ParameterVector`][fatqat.ParameterVector] | One-dimensional NumPy array, or a non-string, non-bytes, non-mapping iterable of accepted scalars | Consumed once in iteration order. The value length must match and every vector element must be used directly as an operation parameter. Bind individual elements for a partial vector. |

[`assign_parameters`][fatqat.Program.assign_parameters] accepts an empty or partial
mapping and returns a new program. It binds only [`Parameter`][fatqat.Parameter]
objects used directly as operation arguments. Unbound parameters remain
symbolic and are rejected by numeric execution and export. String keys,
positional assignments, boolean or complex values, and assigning both a vector
and one of its elements are rejected. Replacement values still undergo the
operation's normal validation. Read [Write quantum computations with Program](../guide/program.md) for the authoring
workflow and [Simulate a quantum program](../guide/simulation.md) for a parameter sweep. The complete
binding and execution contracts are specified here and in [Simulator](simulator.md).

[`copy`][fatqat.Program.copy] and
[`assign_parameters`][fatqat.Program.assign_parameters] return new programs.
[`add`][fatqat.Program.add], [`measure`][fatqat.Program.measure], and
[`measure_all`][fatqat.Program.measure_all] update the current program and return
`None`.

## Draw


FatQat's circuit drawing is based on QuTiP-QIP's circuit drawing tools.
`Program.draw()` translates the Program's instructions to a rendering
adapter before invoking the selected QuTiP-QIP renderer.

**Renderers**

| `renderer` | Return value | Notes |
| --- | --- | --- |
| `"matplotlib"` (default) | Matplotlib `Figure` | Pass `ax=` to draw on an existing axis; other keyword arguments are forwarded to the renderer. |
| `"text"` | Terminal diagram string | The string is returned, not printed. |
| Another QuTiP-QIP renderer name | Renderer-defined | The name and keyword arguments are forwarded unchanged. |

Circuit drawings use one wire per slot but do not depict register dimension.
Unknown or custom operations appear as labeled boxes. A direct
[`PulseOperation`][fatqat.operations.PulseOperation] cannot be represented and raises
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError].

Use
[`fatqat.visualization.to_qubit_circuit`][fatqat.visualization.to_qubit_circuit]
only for low-level integration with QuTiP-QIP's drawing tools:

```python
from fatqat.visualization import to_qubit_circuit

circuit = to_qubit_circuit(program)
```

The returned circuit is a rendering adapter, not an execution object. The old
`fatqat.draw` import path has been removed.

::: fatqat.visualization.to_qubit_circuit

## Reference


::: fatqat.Program
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

## Parameter values


::: fatqat.Parameter
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.ParameterVector
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
