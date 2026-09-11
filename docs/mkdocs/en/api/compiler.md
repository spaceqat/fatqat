---
title: "Compiler"
---

# Compiler

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Pass a [`LogicalProgram`][fatqat.LogicalProgram] to the static gate compiler to
map and lower it for a superconducting or neutral-atom target. Compilation
snapshots the source without editing it. Ordinary [`Program`][fatqat.Program]
instances remain valid simulator inputs but are not accepted by the Python
compiler entry points. See the [compiler guide](../guide/compiler.md) for
supported source behavior and complete execution examples, and see
[LogicalProgram](logical-program.md) for the authoring contract.

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
```

## Compile a Python circuit

::: fatqat.compiler.compile_to_sc

::: fatqat.compiler.compile_to_na

## Compile OpenQASM

::: fatqat.compiler.compile_qasm_to_sc

::: fatqat.compiler.compile_qasm_to_na

## Neutral-atom architecture and visualization

Load a bundled architecture for the NA compiler, then visualize the final
`ZonedPlan` returned in `compiled.output`. Creating an animation requires only
Matplotlib; saving it as MP4 additionally requires FFmpeg on `PATH`.

::: fatqat.compiler.algorithms.load_architecture

::: fatqat.compiler.create_na_animation

::: fatqat.compiler.save_na_animation

## Results

Final target helpers return an executable compilation result. It retains the
compiler IR and pass route while carrying the simulator program and device
layout required by `Simulator.run()`.

::: fatqat.compiler.ExecutableCompilationResult

::: fatqat.compiler.CompilationResult

::: fatqat.ExecutableProgram

## Low-level simulator translation

These functions project manually created or modified final IR. Normal callers
can pass the result of a final-target compile helper directly to the matching
simulator.

::: fatqat.compiler.to_sc_simulator_program

::: fatqat.compiler.to_na_simulator_program

## Explicit pipeline construction

The high-level compile functions above are the normal user interface. These
objects support advanced callers that need to construct or run an explicit
typed pipeline.

::: fatqat.compiler.Compiler

::: fatqat.compiler.CompileContext

::: fatqat.compiler.Pipeline

::: fatqat.compiler.TranslationPass

::: fatqat.compiler.create_sc_pipeline

::: fatqat.compiler.create_na_pipeline

::: fatqat.compiler.IRRegistry

::: fatqat.compiler.IRDefinition

::: fatqat.compiler.IRProgram

## Errors

::: fatqat.compiler.CompilerError

::: fatqat.compiler.ValidationError

::: fatqat.compiler.PassError

::: fatqat.compiler.EmitNotFoundError

::: fatqat.compiler.PipelineNotFoundError

::: fatqat.compiler.UnsupportedFeatureError
