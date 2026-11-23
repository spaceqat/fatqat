---
title: "AtomArraySimulator"
---

# AtomArraySimulator

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] models neutral-atom
arrays at the gate level. Use it when a program's behavior depends on where atoms
have been loaded, which pairs can interact, or whether an atom is lost during a
shot.

## Configure the array { #program-size-mapping-and-native-operations }

The program declares the number of sites. Quantum registers must contain qubits,
and a [`GridRegister`][fatqat.GridRegister] maps to consecutive integer labels,
just like other registers. Use `statevector` or `density_matrix` to evolve the
state.

The native gates are `RX`, `RY`, `RZ`, and `CZ`. To extend them, add rules to the
copy returned by
[`implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map] and
pass it to the constructor.

## Load and pair atoms { #occupancy-and-loss }

Each shot starts with empty sites. Use [`Put`][fatqat.operations.Put] to load an
atom in the ground state or pass an
[`EvolutionState`][fatqat.simulator.EvolutionState] with `classical.occupied` set
to the initially occupied sites. `Put` leaves an occupied site unchanged.

Use unconditional [`Pair`][fatqat.operations.Pair] and
[`Unpair`][fatqat.operations.Unpair] operations to change which sites can run `CZ`
as the program executes. Pairing changes the connection between sites,
independently of whether they currently contain atoms.

A valid gate or reset leaves the state unchanged if a required site is empty. Gate
support and pairing still apply. To model failed loading or subsequent loss,
attach [`Loss`][fatqat.noise.Loss] to the operation where it can occur. For the
rules on other channels, see [Noise
support](../noise/backend-support.md#simulators).

With the Numba runtime and `statevector`, counts-only runs with several shots,
including those with `Put` and `Loss`, accept `shot_parallelism="threads"`.

## Read an empty-site measurement { #example }

Measurement reports `2` for an empty site. This erasure value bypasses readout
noise. The example loads site 0 in the ground state and leaves site 1 empty, so
each shot reports `02`.

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> program = fq.Program(2, 2)
>>> program.add(ops.Put, 0)
>>> program.measure_all()
>>> sim = fq.simulator.AtomArraySimulator(runtime="numpy")
>>> sim.run(program, shots=4).result().get_counts()
{'02': 4}
```

To read which sites still contain atoms, request `final_classical_state` and call
[`Result.get_classical_state()`][fatqat.Result.get_classical_state]. For a final
quantum state from a lossy run, request `final_state=True` with `shots=1`.

## API reference { #api }

<a id="fatqat.simulator.AtomArraySimulator.method"></a>

<a id="fatqat.simulator.AtomArraySimulator.run"></a>

<a id="fatqat.simulator.AtomArraySimulator.run_sweep"></a>

<a id="fatqat.simulator.AtomArraySimulator.validate_noise_model"></a>

Use [`Simulator.run()`][fatqat.simulator.Simulator.run] for a single program and
[`Simulator.run_sweep()`][fatqat.simulator.Simulator.run_sweep] for a parameter
sweep. The entries below cover the simulator-specific interfaces.

::: fatqat.simulator.AtomArraySimulator
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
