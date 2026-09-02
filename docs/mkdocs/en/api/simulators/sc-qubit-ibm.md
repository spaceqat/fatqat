---
title: "SCQubitIBMSimulator"
---

# SCQubitIBMSimulator


[`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] applies the [`Simulator`][fatqat.simulator.Simulator] execution model to
a configurable IBM-style superconducting coupling graph. Use it when native gates,
capacity, and target connectivity matter. It is not a model of an
IBM device and does not transpile, route, schedule, or reproduce a named
processor.

**Hardware profile**

| Property | Value |
| --- | --- |
| Default device | `num_qubits=16`; a 4×4 reference coupling graph |
| Uniform native gates | [`fatqat.operations.X`][fatqat.operations.X], [`fatqat.operations.SX`][fatqat.operations.SX], [`fatqat.operations.RZ`][fatqat.operations.RZ] |
| Connected native gate | [`fatqat.operations.CZ`][fatqat.operations.CZ] on the supplied couplings |
| Other built-in operations | Measurement and [`fatqat.operations.Reset`][fatqat.operations.Reset] follow the method rules described by [`Simulator`][fatqat.simulator.Simulator] |
| Methods | All methods supported by [`Simulator`][fatqat.simulator.Simulator]; default `statevector` |
| Runtime | `numba` by default; `numpy` is also supported |
| Noise | Ideal by default; a built-in reference model is available explicitly |

## Native gates and layout


The default row-major numbering is:

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

Both operand orders of every coupling are legal, so `CZ` is accepted on
device labels `(0, 1)` and `(1, 0)`, but not `(0, 5)` in the default data.
The grid is only the default test topology; callers may provide any valid
undirected coupling graph.

With the automatic layout, a program maps all quantum registers to device labels
`0, 1, ...` in declaration order. A [`GridRegister`][fatqat.GridRegister] is
flattened in the same order as any other register; frontend geometry does not
define hardware placement. An explicit, complete
[`ResourceLayout`][fatqat.ResourceLayout] can place program references differently.
Capacity and the qubit-only restriction still apply.

The backend executes only programs already written in its native gate set.
For example, [`fatqat.operations.CX`][fatqat.operations.CX] is rejected even on neighbouring
qubits. Inspect [`SCQubitIBMSimulator.implementation_map`][fatqat.simulator.SCQubitIBMSimulator.implementation_map] instead of
hard-coding these rules:

```python
import fatqat as fq
import fatqat.operations as ops

backend = fq.simulator.SCQubitIBMSimulator(
    num_qubits=5,
    couplings=((0, 1), (1, 2), (1, 3), (3, 4)),
)
native = backend.implementation_map

assert native.supports(ops.SX)
assert native.supports(ops.CZ, device_operands=(0, 1))
assert not native.supports(ops.CZ, device_operands=(0, 3))
```

`device_operands_for(operation)` returns an empty set for a gate available
everywhere and ordered tuples for a connectivity-limited gate.

## Built-in noise


The simulator remains ideal unless a noise model is supplied. To use the
built-in profile, request it explicitly:

```python
import fatqat as fq

Sim = fq.simulator.SCQubitIBMSimulator
backend = Sim(noise=Sim.default_noise_model())
```

The profile uses `T1 = 60 us` and `T2 = 48 us`. Each call returns a fresh
[`NoiseModel`][fatqat.NoiseModel] that you can extend before passing it to the
simulator.

**Built-in profile**

| Operation | Duration | Noise |
| --- | --- | --- |
| `X`, `SX` | 20 ns | T1/T2-derived amplitude and phase damping |
| `RZ` | 0 ns (virtual) | None |
| `CZ` | 50 ns | Relaxation on each qubit, followed by joint depolarizing noise with `p = 0.001` |
| Measurement | — | `P(report 1 \| true 0) = 0.02` and `P(report 0 \| true 1) = 0.04` |

See [Noise](../noise.md) for how the selected simulation method applies this model.

## API


[`Simulator.method`][fatqat.simulator.Simulator.method], [`SCQubitIBMSimulator.implementation_map`][fatqat.simulator.SCQubitIBMSimulator.implementation_map],
[`Simulator.run`][fatqat.simulator.Simulator.run], [`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep], and
[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] follow the general Simulator API and are
included below for a complete class reference.

::: fatqat.simulator.SCQubitIBMSimulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
