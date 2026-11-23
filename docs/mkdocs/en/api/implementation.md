---
title: "Matrix implementation maps"
---

# Matrix implementation maps { #implementation-maps }

When a simulator encounters a gate, its implementation map determines which matrix
to apply. With
[`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap], you
can add a rule, change a built-in one, or use different matrices for the same gate
at different device locations. Pass the map as `implementation_map` when you
create the simulator. The map also supplies channel rules. The `NoiseModel`
determines when a channel acts and which targets it affects.

## Choose a starting map { #choose-a-map }

To keep the built-in gate and channel rules, start with
[`default_matrix_implementation_map()`][fatqat.implementation.default_matrix_implementation_map].
To provide every rule yourself, create an empty
[`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap]. The
simulator uses a copy of the registrations made before it was created. Additions
and removals you make later affect only the original map.

## Choose a rule

Register a rule with
[`MatrixImplementationMap.add()`][fatqat.implementation.MatrixImplementationMap.add].
You can pass an operation or channel instance, or its class. For an operation
implemented with a matrix, choose one of the following forms.

| Form | Use it for | Value passed to `add` |
| --- | --- | --- |
| Two-dimensional NumPy array | One constant matrix | The array; FATQAT copies it, so later changes do not affect the rule |
| Callable | A matrix that depends on operation parameters or target dimensions | `rule(op)` by default. If the callable accepts a `targets=` keyword or `**kwargs`, FATQAT calls `rule(op, targets=targets)`. |
| [`MatrixImplementation`][fatqat.implementation.MatrixImplementation] | A configured or stateful rule object | Override `__call__(op, *, targets)`. |

Use a callable that returns Kraus matrices for a [custom
channel](noise/custom-implementations.md). To update the quantum and classical
state together, register a `CustomOperationImplementation`. See [Custom
operations](simulators/custom-operations.md) for that interface.

In a matrix rule, `op` is the operation instance, including any parameters such as
a rotation angle. `targets` gives one [`RegisterRef`][fatqat.RegisterRef] per
operand, in program order. Read `target.register.dim` if the matrix depends on a
target's dimension.

## Local basis order

Matrix factors follow the target tuple passed to
[`add`][fatqat.Program.add]. The first target is the most-significant local
factor and the last target changes fastest. For dimensions `(d0, d1, ..., dk)`
and basis digits `(b0, b1, ..., bk)`, the flat local index is
`b0 * d1 * ... * dk + b1 * d2 * ... * dk + ... + bk`.

For two qubits, targets `(q0, q1)` therefore use the local basis `|00>`,
`|01>`, `|10>`, `|11>`. Controlled operations list controls before
targets. When the tuple lists every subsystem in public order, this is also
the basis used by returned states and operators. A local matrix on a smaller
or differently ordered target tuple must be embedded before direct use with a
full returned state; the simulator performs that embedding during execution.

## Target-specific rules

Pass `device_operands=` to `add` to register a rule only for that exact
ordered tuple of backend-defined physical labels. These labels are not program
register references. One operation family uses exactly one registration mode:

- A *uniform* rule, added without `device_operands`, matches every physical
  tuple of the correct arity.
- *Device-specific* rules match only their explicitly registered tuples. You
  may add several tuples for the same operation.

Call [`remove`][fatqat.implementation.MatrixImplementationMap.remove] before
switching a family between these modes.

Both modes accept `num_subsystems=None` and instance-dependent channel widths.
Their device-specific tuples may have different lengths. A fixed integer
`num_subsystems` requires that many labels.

## Registration and run errors { #validation-timing }

[`add()`][fatqat.implementation.MatrixImplementationMap.add] checks the
registration when you add it. A rule that raises while the backend prepares an
operation can produce
[`MatrixImplementationError`][fatqat.errors.MatrixImplementationError]. A matrix
with the wrong shape raises
[`BackendValidationError`][fatqat.errors.BackendValidationError], and an
unregistered operation raises
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError].

## API reference { #reference }

### data `fatqat.implementation.DeviceOperands` { #fatqat.implementation.DeviceOperands }

Alias for an ordered tuple of hashable backend-defined physical labels.

::: fatqat.implementation.default_matrix_implementation_map

::: fatqat.implementation.MatrixImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.implementation.MatrixImplementation
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.implementation.FixedMatrix
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
