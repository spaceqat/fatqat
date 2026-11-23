---
title: "Compiler pipelines & IR"
---

# Compiler pipelines & IR

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Use the pipeline API to run a custom sequence of compiler passes or inspect
its intermediate representations. To compile a circuit for a supported
device, use the entry points on the [Compiler](compiler.md) page.

<a id="pipelines-and-irs"></a>

## Run a pipeline

Start with a registered superconducting or neutral-atom pipeline. Call
[`Compiler.compile()`][fatqat.compiler.Compiler.compile] with the pipeline name
and a [`CompileContext`][fatqat.compiler.CompileContext] for the target and any
options. Use `emit` to stop at a particular representation.

::: fatqat.compiler.create_sc_pipeline
    options:
      heading_level: 3

::: fatqat.compiler.create_na_pipeline
    options:
      heading_level: 3

::: fatqat.compiler.Compiler
    options:
      heading_level: 3

::: fatqat.compiler.CompileContext
    options:
      heading_level: 3

<a id="simulator-translation"></a>

## Convert to a simulator program

Use these functions when you have a final target representation, for example
the `output` from calling `Compiler.compile()` on a custom pipeline. The
conversion returns a simulator program and its resource layout. The four
compile functions already do this conversion for their default results.

::: fatqat.compiler.dialects.SCNativeProgram
    options:
      heading_level: 3

::: fatqat.compiler.to_sc_simulator_program
    options:
      heading_level: 3

::: fatqat.compiler.to_na_simulator_program
    options:
      heading_level: 3

## Define passes and IRs

Each pass converts one registered representation into another. A pipeline
lists those passes in order: each pass must accept the representation produced
by the previous one.

::: fatqat.compiler.Pipeline
    options:
      heading_level: 3

::: fatqat.compiler.TranslationPass
    options:
      heading_level: 3

::: fatqat.compiler.IRRegistry
    options:
      heading_level: 3

::: fatqat.compiler.IRDefinition
    options:
      heading_level: 3

::: fatqat.compiler.IRProgram
    options:
      heading_level: 3
