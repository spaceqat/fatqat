---
title: "Compiler"
---

# Compiler

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Use the compiler to prepare a device-independent circuit for a superconducting
or neutral-atom target. A final compilation can be passed directly to the
matching simulator.

Choose a function based on your input and target:

| Source | Superconducting target | Neutral-atom target |
| --- | --- | --- |
| [`LogicalProgram`](logical-program.md) | [`compile_to_sc()`][fatqat.compiler.compile_to_sc] | [`compile_to_na()`][fatqat.compiler.compile_to_na] |
| OpenQASM 2 or 3 | [`compile_qasm_to_sc()`][fatqat.compiler.compile_qasm_to_sc] | [`compile_qasm_to_na()`][fatqat.compiler.compile_qasm_to_na] |

For superconducting compilation, pass an [`SCTarget`](sc-target.md) or an
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator]. For neutral-atom
compilation, pass an architecture such as one returned by
[`load_architecture()`][fatqat.compiler.algorithms.load_architecture].

## Compile a LogicalProgram

::: fatqat.compiler.compile_to_sc
    options:
      heading_level: 3

::: fatqat.compiler.compile_to_na
    options:
      heading_level: 3

## Compile OpenQASM

Pass OpenQASM source text directly to the target compiler. The returned result
uses the same execution interface as a compiled `LogicalProgram`.

::: fatqat.compiler.compile_qasm_to_sc
    options:
      heading_level: 3

::: fatqat.compiler.compile_qasm_to_na
    options:
      heading_level: 3

## Compilation results

With the default `emit`, the four compile functions return an
[`ExecutableCompilationResult`][fatqat.compiler.ExecutableCompilationResult].
Pass it directly to a compatible simulator. To inspect an intermediate
representation, set `emit` to the boundary you want and read `output` from the
returned [`CompilationResult`][fatqat.compiler.CompilationResult].

::: fatqat.compiler.ExecutableCompilationResult
    options:
      heading_level: 3

::: fatqat.compiler.CompilationResult
    options:
      heading_level: 3

<a id="fatqat.ExecutableProgram"></a>
<a id="fatqat.ExecutableProgram.program"></a>
<a id="fatqat.ExecutableProgram.resource_layout"></a>

For the protocol used by simulators, see
[`ExecutableProgram`][fatqat.ExecutableProgram] in the simulator reference.

## Neutral-atom architectures and animation

`load_architecture()` returns a fresh copy of a bundled neutral-atom profile.
Use the same architecture for compilation and schedule visualization.

::: fatqat.compiler.algorithms.load_architecture
    options:
      heading_level: 3

::: fatqat.compiler.create_na_animation
    options:
      heading_level: 3

::: fatqat.compiler.save_na_animation
    options:
      heading_level: 3

## Errors

Compiler errors share the `CompilerError` base class. Check
`PassError.pass_name` to find the failing stage. If `__cause__` is set, it
contains the original exception.

::: fatqat.compiler.CompilerError
    options:
      heading_level: 3

::: fatqat.compiler.ValidationError
    options:
      heading_level: 3

::: fatqat.compiler.PassError
    options:
      heading_level: 3

::: fatqat.compiler.EmitNotFoundError
    options:
      heading_level: 3

::: fatqat.compiler.PipelineNotFoundError
    options:
      heading_level: 3

::: fatqat.compiler.UnsupportedFeatureError
    options:
      heading_level: 3

## Advanced compiler APIs

<a id="pipelines-and-irs"></a>
<a id="simulator-translation"></a>
<a id="fatqat.compiler.dialects.SCNativeProgram"></a>
<a id="fatqat.compiler.to_sc_simulator_program"></a>
<a id="fatqat.compiler.to_na_simulator_program"></a>
<a id="fatqat.compiler.Compiler"></a>
<a id="fatqat.compiler.CompileContext"></a>
<a id="fatqat.compiler.Pipeline"></a>
<a id="fatqat.compiler.TranslationPass"></a>
<a id="fatqat.compiler.create_sc_pipeline"></a>
<a id="fatqat.compiler.create_na_pipeline"></a>
<a id="fatqat.compiler.IRRegistry"></a>
<a id="fatqat.compiler.IRDefinition"></a>
<a id="fatqat.compiler.IRProgram"></a>

See [Compiler pipelines & IR](compiler-pipelines.md) to configure compiler
passes, inspect intermediate representations, or
[convert a final representation for simulation](compiler-pipelines.md#simulator-translation).
