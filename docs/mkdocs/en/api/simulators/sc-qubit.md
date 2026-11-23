---
title: "SCQubitSimulator"
---

# SCQubitSimulator

[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] runs gate-level programs
on a configurable superconducting qubit layout. Use it to check whether a program
fits the available qubit sites, uses native gates, and applies `CZ` only between
connected sites.

## Configure layout and noise { #native-gates-and-layout }

Choose the number of qubit sites and which pairs can interact. The default is a
4×4 nearest-neighbor grid, with sites numbered from 0 to 15 in row order. To use
another graph, set `num_qubits` and `couplings`.

By default, qubits map to consecutive device labels in register declaration order,
including the members of a [`GridRegister`][fatqat.GridRegister]. Pass a complete
[`ResourceLayout`][fatqat.ResourceLayout] to place those qubits at other sites. To
compile a logical circuit for the same layout, pass the target returned by
[`get_compiler_target()`][fatqat.simulator.SCQubitSimulator.get_compiler_target]
to the compiler.

### Built-in noise { #built-in-noise }

To include gate damping and readout errors, pass the model returned by
[`default_noise_model()`][fatqat.simulator.SCQubitSimulator.default_noise_model]
as `noise`. Its API entry lists the gate and readout parameters.

## Check native gates { #check-native-gates }

The simulator accepts `X`, `SX`, and `RZ` at every site. A `CZ` accepts either
order of a connected pair, such as `(0, 1)` or `(1, 0)` on the default grid.
Measurement and reset follow the [rules for the chosen simulation
method](../simulator.md#methods).

To check a pair before running a program, query
[`implementation_map`][fatqat.simulator.SCQubitSimulator.implementation_map]. On
the default grid, sites 0 and 5 are diagonal neighbors, so this pair cannot run
`CZ`.

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> sim = fq.simulator.SCQubitSimulator(runtime="numpy")
>>> sim.implementation_map.supports(ops.CZ, device_operands=(0, 5))
False
```

## API reference { #api }

<a id="fatqat.simulator.SCQubitSimulator.method"></a>

<a id="fatqat.simulator.SCQubitSimulator.run"></a>

<a id="fatqat.simulator.SCQubitSimulator.run_sweep"></a>

<a id="fatqat.simulator.SCQubitSimulator.validate_noise_model"></a>

Use [`Simulator.run()`][fatqat.simulator.Simulator.run] for a single program and
[`Simulator.run_sweep()`][fatqat.simulator.Simulator.run_sweep] for a parameter
sweep. The entries below cover the simulator-specific interfaces.

::: fatqat.simulator.SCQubitSimulator
    options:
      inherited_members:
        - implementation_map
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
