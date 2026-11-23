---
title: "SCTarget"
---

# SCTarget

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Use [`SCTarget`][fatqat.compiler.SCTarget] to tell the compiler how many sites
the superconducting device has and which pairs can run `CZ` directly. Pass it
to [`compile_to_sc()`][fatqat.compiler.compile_to_sc] or
[`compile_qasm_to_sc()`][fatqat.compiler.compile_qasm_to_sc] in place of a
simulator.

<a id="why-compile-against-a-target"></a>
<a id="compiled-output"></a>
<a id="when-errors-are-raised"></a>

## Target and simulator compatibility

An `SCTarget` specifies the topology. The compiler uses the superconducting
native gates `X`, `SX`, `RZ`, and `CZ`, with `CZ` limited to the target's
couplings. The result can run on any compatible simulator that provides
the sites and couplings it uses and supports the program with the selected
simulation method.

Pass an [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] directly when
its implementation map should also constrain routing and emitted gates.
To reuse just its topology, call
[`get_compiler_target()`][fatqat.simulator.SCQubitSimulator.get_compiler_target].

<a id="construct-a-target"></a>
<a id="compile-and-run"></a>

## API reference { #reference }

::: fatqat.compiler.SCTarget
    options:
      heading_level: 3
      merge_init_into_class: true
