# Simulate a neutral-atom array

A neutral-atom array uses individual atoms as qubits. Alongside the quantum
state, it matters whether an atom is present at each site and which atoms can
interact. Loading, pairing, and losing atoms can change which operations are
available as the program runs.

Use [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] to track atom
loading, pairing, and loss while running circuit operations. It does not
integrate a Hamiltonian or calculate physical atom trajectories.

## Track atom occupancy and pairing { #atom-occupancy-and-pairing }

The atom array has no fixed geometry.
Program resources define sites that begin empty. `Put` loads
the atoms, while `Pair` and `Unpair` reshape the connectivity on which `CZ`
is legal:

![Two occupied atoms begin separated, move together when Pair applies with depolarizing noise, remain paired for CZ, and separate again when Unpair applies with depolarizing noise.](../assets/generated/guide/atom-pairing-lifecycle.svg)

In the diagram, bringing the atoms together represents creating a pair. The
program below loads an atom at each site and then uses `Pair` to make `CZ`
available on those sites. `Unpair` removes that connection.

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> atoms = fq.Program(2, 2)
>>> atoms.add(ops.Put, (0, 1))
>>> atoms.add(ops.Pair, (0, 1))
>>> atoms.add(ops.RX(np.pi), 0)
>>> atoms.add(ops.CZ, (0, 1))
>>> atoms.add(ops.Unpair, (0, 1))
>>> atoms.measure_all()
>>> atom_counts = fq.simulator.AtomArraySimulator().run(
...     atoms,
...     shots=8,
...     execution_config={"seed": 7},
... ).result().get_counts()
>>> atom_counts
{'10': 8}
```

Pairing is ideal unless you attach a noise assumption. For example, apply a
small depolarizing channel independently to each atom whenever either movement
instruction occurs:

```pycon
>>> movement_noise = fq.NoiseModel()
>>> for movement in (ops.Pair, ops.Unpair):
...     for target_position in (0, 1):
...         movement_noise.add(
...             fq.noise.Depolarizing(p=0.02),
...             operation=movement,
...             target_positions=target_position,
...         )
>>> noisy_atom_backend = fq.simulator.AtomArraySimulator(
...     noise=movement_noise,
... )
>>> noisy_counts = noisy_atom_backend.run(
...     atoms,
...     shots=100,
...     execution_config={"seed": 7},
... ).result().get_counts()
>>> sum(noisy_counts.values())
100
```

This channel perturbs the quantum state during `Pair` and `Unpair`; it does not
remove either atom. Use [`Loss`][fatqat.noise.Loss] instead when movement
should change occupancy.

![An occupied atom undergoes a matched operation, after which Loss either leaves it present with probability one minus p or removes it with probability p; measurement of the empty site returns 2.](../assets/generated/guide/atom-loss-lifecycle.svg)

Loss can be attached to any supported operation. Here it is sampled after `RX`,
so surviving atoms return `1`, while lost atoms return `2`:

```pycon
>>> loss_model = fq.NoiseModel()
>>> loss_model.add(fq.noise.Loss(p=0.1), operation=ops.RX)
>>> lossy_atoms = fq.Program(1, 1)
>>> lossy_atoms.add(ops.Put, 0)
>>> lossy_atoms.add(ops.RX(np.pi), 0)
>>> lossy_atoms.measure_all()
>>> lossy_counts = fq.simulator.AtomArraySimulator(noise=loss_model).run(
...     lossy_atoms,
...     shots=100,
...     execution_config={"seed": 7},
... ).result().get_counts()
>>> lossy_counts
{'1': 86, '2': 14}
```
Omitting `Pair` before `CZ` is a program error. A missing atom is different:
supported gates find nothing to act on, and measurement reports the erasure
digit `2`.

For native gates, occupancy rules, and simulation methods, see the
[atom-array simulator API](../api/simulators/atom-array.md). For pulse-driven
physical dynamics, see [Neutral-atom emulation](neutral-atom-emulation.md).
