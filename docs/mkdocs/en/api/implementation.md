---
title: "Implementation maps"
---

# Implementation maps


Use [`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap] to add custom
operations and channels to a simulator, or choose rules by physical target. Pass
the map as `implementation_map=` when constructing the backend. Built-in
gates and channels work without a custom map. Noise placement is still defined
by `NoiseModel`; the map supplies how each matched channel acts.

## Choose a map


[`default_matrix_implementation_map`][fatqat.implementation.default_matrix_implementation_map] returns a
fresh map containing FATQAT's built-in matrix gates and finite channel rules. Editing it does not affect
another map or a later call. Construct
[`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap] directly when the
backend should support only rules you add. A supplied map replaces both the
gate and channel registrations; start with the default map to retain built-ins.

## Choose a rule


[`MatrixImplementationMap.add`][fatqat.implementation.MatrixImplementationMap.add] accepts an operation
or channel instance or class. Matrix operations accept these rule forms:

**Matrix rule forms**

| Form | Use it for | Value passed to `add` |
| --- | --- | --- |
| Two-dimensional NumPy array | One constant matrix | The array; FATQAT copies it, so later changes do not affect the rule |
| Callable | A matrix that depends on operation parameters or target dimensions | `rule(op)` by default. If the callable accepts a `targets=` keyword or `**kwargs`, FATQAT calls `rule(op, targets=targets)`. |
| [`MatrixImplementation`][fatqat.implementation.MatrixImplementation] | A configured or stateful rule object | Override `__call__(op, *, targets)`. |

Channel rules are callables returning Kraus matrices; see
[Custom noise implementations](noise/custom-implementations.md). For operations
that update quantum and classical state together, the same map
accepts `CustomOperationImplementation` objects. See
[Custom operations](simulators/custom-operations.md).

`op` is the applied operation value, so it contains parameters such as a
rotation angle. `targets` contains scalar program
[`RegisterRef`][fatqat.RegisterRef] objects in operand order; a rule can inspect
`target.register.dim` when its matrix depends on subsystem dimension.

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

## Validation timing


`add` rejects an invalid registration immediately with `TypeError` or
`ValueError`. Matrix and channel rules run when a backend prepares a program. A matrix rule that
raises produces [`MatrixImplementationError`][fatqat.errors.MatrixImplementationError], an array
with the wrong target shape produces
[`BackendValidationError`][fatqat.errors.BackendValidationError], and a missing rule produces
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]. These errors occur before
`run` returns a [`Job`][fatqat.Job].

## Reference


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
