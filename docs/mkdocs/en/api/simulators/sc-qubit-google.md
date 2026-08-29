---
title: "SCQubitGoogleSimulator"
---

# SCQubitGoogleSimulator


[`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] applies the [`Simulator`][fatqat.simulator.Simulator] execution model
to a configurable Google-style superconducting grid. Use it to test native
rotation gates and nearest-neighbour `iSwap`/`CZ` programs. It is not a
model of a named Google processor and does not transpile, route, schedule, or
reproduce hardware calibration data.

**Hardware profile**

| Property | Value |
| --- | --- |
| Default device | `grid_size=(4, 4)`; 16 row-major qubits |
| Uniform native gates | [`fatqat.operations.RX`][fatqat.operations.RX], [`fatqat.operations.RY`][fatqat.operations.RY], [`fatqat.operations.RZ`][fatqat.operations.RZ] |
| Connected native gates | [`fatqat.operations.iSwap`][fatqat.operations.iSwap] and [`fatqat.operations.CZ`][fatqat.operations.CZ] on horizontal or vertical neighbours |
| Other built-in operations | Measurement and [`fatqat.operations.Reset`][fatqat.operations.Reset] follow the method rules described by [`Simulator`][fatqat.simulator.Simulator] |
| Methods | All methods supported by [`Simulator`][fatqat.simulator.Simulator]; default `statevector` |
| Runtime | `numba` by default; `numpy` is also supported |
| Noise | Ideal by default; a built-in reference model is available explicitly |

## Native gates and layout


Device labels are row-major. On the default grid they are:

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

Both operand orders of every grid edge are legal. Thus `iSwap` is accepted
on device labels `(0, 1)` and `(1, 0)`, but not `(0, 5)`. `CZ` follows
the same rule for any positive `grid_size=(rows, columns)`.

With the automatic layout, an ordinary program maps its qubits to device labels
`0, 1, ...` in declaration order. One [`GridRegister`][fatqat.GridRegister] instead
maps into the device's top-left corner while preserving row and column
coordinates. In this automatic mode, it must be the program's only quantum
register and must fit along both device axes. An explicit, complete
[`ResourceLayout`][fatqat.ResourceLayout] can place program references differently.
Capacity and the qubit-only restriction still apply.

The backend does not decompose non-native operations. In particular,
[`fatqat.operations.CX`][fatqat.operations.CX] and [`fatqat.operations.SX`][fatqat.operations.SX] are rejected.
Inspect [`SCQubitGoogleSimulator.implementation_map`][fatqat.simulator.SCQubitGoogleSimulator.implementation_map] to see the gate set
and legal operand tuples:

```python
import fatqat as fq
import fatqat.operations as ops

backend = fq.simulator.SCQubitGoogleSimulator(grid_size=(2, 3))
native = backend.implementation_map

assert native.supports(ops.RY)
assert native.supports(ops.iSwap, device_operands=(1, 4))
assert not native.supports(ops.iSwap, device_operands=(0, 4))
```

`device_operands_for(operation)` returns an empty set for a gate available
everywhere and ordered tuples for a connectivity-limited gate.

## Built-in noise


The simulator remains ideal unless a noise model is supplied. To use the
built-in profile, request it explicitly:

```python
import fatqat as fq

Sim = fq.simulator.SCQubitGoogleSimulator
backend = Sim(noise=Sim.default_noise_model())
```

The profile uses `T1 = 60 us` and `T2 = 48 us`. Each call returns a fresh
[`NoiseModel`][fatqat.NoiseModel] that you can extend before passing it to the
simulator.

**Built-in profile**

| Operation | Duration | Noise |
| --- | --- | --- |
| `RX`, `RY`, `RZ` | 20 ns | T1/T2-derived amplitude and phase damping |
| `iSwap` | 30 ns | Relaxation on each qubit, followed by joint depolarizing noise with `p = 0.001` |
| `CZ` | 50 ns | Relaxation on each qubit, followed by joint depolarizing noise with `p = 0.001` |
| Measurement | — | `P(report 1 \| true 0) = 0.02` and `P(report 0 \| true 1) = 0.04` |

Unlike the IBM-style profile, `RZ` is a physical, noisy 20 ns rotation in
this profile. See [Noise](../noise.md) for method-dependent channel execution.

## API


[`Simulator.method`][fatqat.simulator.Simulator.method], [`SCQubitGoogleSimulator.implementation_map`][fatqat.simulator.SCQubitGoogleSimulator.implementation_map],
[`Simulator.run`][fatqat.simulator.Simulator.run], [`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep], and
[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] follow the general Simulator API and are
included below for a complete class reference.

::: fatqat.simulator.SCQubitGoogleSimulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
