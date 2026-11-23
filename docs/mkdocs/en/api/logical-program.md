---
title: "LogicalProgram"
---

# LogicalProgram

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

A [`LogicalProgram`][fatqat.LogicalProgram] describes a device-independent
circuit. Run it directly with the [general simulator](simulator.md), or pass it
to the [compiler](compiler.md#compile-a-logicalprogram) for a superconducting
or neutral-atom target. Use [`Program`](program.md) for operations that address
hardware directly, such as atom placement, pairing, or pulse controls, and
for custom operations.

A circuit can be valid for simulation without being valid for compilation.
For example, the general simulator can run a circuit with classical conditions.
The current compiler requires a static circuit with measurements at the end
and no classical conditions. Bind any symbolic parameters before compiling.
The supported gates also depend on the target.

<a id="compile-or-simulate-the-circuit"></a>
<a id="choose-between-logicalprogram-and-program"></a>
<a id="reuse-parameterized-circuits"></a>

## API reference { #reference }

`LogicalProgram` inherits its [constructor][fatqat.Program],
[`measure()`][fatqat.Program.measure], [`measure_all()`][fatqat.Program.measure_all],
and [`draw()`][fatqat.Program.draw] from [`Program`][fatqat.Program].

::: fatqat.LogicalProgram
    options:
      heading_level: 3
      inherited_members: false
      show_bases: true
