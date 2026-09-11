---
title: "LogicalProgram"
---

# LogicalProgram

[`LogicalProgram`][fatqat.LogicalProgram] is a restricted
[`Program`](program.md) for device-independent circuits. It uses the same
constructor and authoring interface; `add()`, `measure()`, and `measure_all()`
mutate in place and return `None`. Use LogicalProgram, rather than its Program
base class, with `compile_to_sc()` and `compile_to_na()`.

`add()` accepts the built-in qubit and qudit circuit gates, reset, and
barriers. It rejects device operations (including atom placement, pairing,
and pulse controls) and custom operation classes, including subclasses of
built-in gates, with `ValueError`. Measurements use `measure()`.

Classical conditions remain available for direct simulation. Register views
follow `Program`'s authoring rules and expand into scalar gates when compiled.
Compilation applies the
[static gate and target restrictions](../guide/compiler.md#supported-source-behavior).

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2, metadata={"name": "bell"})
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
compiled = fq.compiler.compile_to_sc(circuit, fq.simulator.SCQubitSimulator())
```

The `copy()` and `assign_parameters()` methods return independently editable
`LogicalProgram` instances with the same operation restrictions. Their public
return annotations also preserve `LogicalProgram` for static type checkers and
editor completion. See [Compiler](compiler.md) for compilation options and
results.

## Reference

::: fatqat.LogicalProgram
    options:
      inherited_members: false
      show_bases: true
