---
title: "Compiler"
---

# Compiler

Use the compiler to prepare a device-independent circuit for a superconducting
or neutral-atom target. A final compilation can be passed directly to the
matching simulator.

Choose an entry point from the source and target you have:

| Source | Superconducting target | Neutral-atom target |
| --- | --- | --- |
| [`LogicalProgram`](logical-program.md) | `compile_to_sc()` | `compile_to_na()` |
| OpenQASM 2 or 3 | `compile_qasm_to_sc()` | `compile_qasm_to_na()` |

The [compiler guide](../guide/compiler.md) shows complete compile-and-run
examples for all four paths.

## Compile a LogicalProgram

Build a circuit with `LogicalProgram`, then provide either an SC backend or a
neutral-atom architecture:

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()

backend = fq.simulator.SCQubitSimulator()
compiled = fq.compiler.compile_to_sc(circuit, backend)
result = backend.run(compiled, shots=100).result()
```

::: fatqat.compiler.compile_to_sc

::: fatqat.compiler.compile_to_na

## Compile OpenQASM

Pass OpenQASM source text directly to the target compiler. The returned result
uses the same execution interface as a compiled `LogicalProgram`.

::: fatqat.compiler.compile_qasm_to_sc

::: fatqat.compiler.compile_qasm_to_na

## Compilation results

At the final target boundary, the compile functions return an
`ExecutableCompilationResult`. Pass it to the matching simulator, inspect the
target representation through `.output`, or review the stages through
`.route`.

When `emit` selects an earlier boundary, the return value is a
`CompilationResult` intended for inspection.

::: fatqat.compiler.ExecutableCompilationResult

::: fatqat.compiler.CompilationResult

::: fatqat.ExecutableProgram

## Neutral-atom architectures and animation

`load_architecture()` returns a fresh copy of a bundled neutral-atom profile.
Use the same architecture for compilation and schedule visualization.

::: fatqat.compiler.algorithms.load_architecture

::: fatqat.compiler.create_na_animation

::: fatqat.compiler.save_na_animation

## Advanced compiler APIs

Most applications only need the four compile functions above. The following
APIs expose intermediate translation and pipeline construction for compiler
extensions and IR inspection.

### Simulator translation

Use these functions when working with a final compiler IR that was constructed
or modified directly. Results returned by the normal compile functions are
already ready for simulation.

SC native programs can declare `classical_registers` as a keyword-only tuple
of the original register objects in output order. The bridge retains all slots,
including unwritten slots and entirely unused registers. Distinct registers may
share a name; declaring the same object twice is invalid. Measurement outputs
must belong to declared registers. The bridge and native verifier raise
`ValidationError` for invalid explicit declarations.

The default `None` preserves compatibility with native programs constructed
without declarations: the bridge discovers registers in first measurement
occurrence order. This fallback cannot recover original declaration order or
registers with no measurements. Use `()` to declare no classical registers;
it does not request inference. Compiler-generated native programs always carry
explicit declarations. Adding this field preserves the existing three positional
constructor arguments; it also participates in dataclass equality and reflection.

::: fatqat.compiler.dialects.SCNativeProgram

::: fatqat.compiler.to_sc_simulator_program

::: fatqat.compiler.to_na_simulator_program

### Pipelines and IRs

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

Compiler errors share the `CompilerError` base class. `PassError` includes the
stage that failed and preserves the original error as its cause.

::: fatqat.compiler.CompilerError

::: fatqat.compiler.ValidationError

::: fatqat.compiler.PassError

::: fatqat.compiler.EmitNotFoundError

::: fatqat.compiler.PipelineNotFoundError

::: fatqat.compiler.UnsupportedFeatureError
