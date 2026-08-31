# Choose and run a neutral-atom workflow

FatQat offers two neutral-atom execution levels. Choose the least detailed
one that still contains the effect you want to study:

<div class="grid cards" markdown>

-   **Gate-level array**

    [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] adds occupancy and a changing
    pairing graph to finite qubit gates. Use it for fast loading, loss, and
    connectivity checks.

-   **Two physical levels**

    [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] follows $\lvert g\rangle$ and
    $\lvert r\rangle$. Use it for global drive, detuning, and many-body Rydberg
    dynamics.

</div>

The atom-array simulator is the profile introduced in
[Hardware-profile simulation](hardware-profile-simulation.md); it does not
integrate a Hamiltonian. The physical emulator below uses the pulse workflow
from [Hamiltonian emulation](hamiltonian-emulation.md).

## Describe the sites once

The physical emulator uses a fixed set of sites described by an
[`AtomArrangement`][fatqat.emulator.AtomArrangement]. Program resources map to those
sites in declaration order by default:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> arrangement = fq.emulator.AtomArrangement.chain(
...     num_sites=2,
...     spacing=6.0,
... )
>>> arrangement.num_sites
2
```

The arrangement is fixed geometry, not an atom-transport instruction. It sets
the distances used by Rydberg interactions, and the Program must declare
exactly one dimension-two resource per site.

Use a [`ResourceLayout`][fatqat.ResourceLayout] only when you need to override the
default declaration-order mapping.

## Drive the array with the two-level model

The two-level model uses global drive and detuning channels. Its default gate
map is empty, so a Program normally contains direct pulse blocks:

```pycon
>>> atom2_model = fq.emulator.Atom2LevelModel.from_document(
...     fq.emulator.load_model_document("atom2level.reference")
... )
>>> atom2 = fq.emulator.Atom2LevelEmulator(
...     atom2_model,
...     arrangement=arrangement,
... )
>>> drive = fq.emulator.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (0.0, 0.5, 0.0),
... )
>>> detuning = fq.emulator.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (0.0, 0.1, 0.0),
... )
>>> global_controls = (
...     fq.emulator.PulseControl(atom2_model.control.drive(), drive),
...     fq.emulator.PulseControl(atom2_model.control.detuning(), detuning),
... )
>>> global_program = fq.Program(arrangement.num_sites)
>>> global_program.add(ops.PulseOperation(1.0, global_controls))
>>> atom2_state = atom2.run(global_program).result().get_statevector()
>>> atom2_state.shape
(4,)
>>> bool(np.isclose(np.linalg.norm(atom2_state), 1.0))
True
>>> round(float(1.0 - abs(atom2_state[0]) ** 2), 3)
0.054
```

Both controls act on every site. The arrangement supplies the distances and
the model supplies the signed interaction strength, so changing the spacing
can change the dynamics without changing the Program. By default every site
pair interacts. Set `simulation_config={"interaction_cutoff": distance}` on a
particular `run()` to remove terms beyond that distance; this is a numerical
Hamiltonian truncation rather than a blockade radius.

For this short pulse, about 5.4% of the final probability lies outside the
all-ground state. That number is a physical consequence of the waveform,
detuning, spacing, interaction model, and duration—not a gate label.

The default method returns the complete two-level statevector. With supported
Lindblad noise it follows one seeded trajectory when a final state is
explicitly requested; choose `method="density_matrix"` for the exact ensemble.
Terminal measurement returns counts. Mid-circuit measurement, reset,
conditions, and per-site direct controls are not part of this two-level
workflow.

## Continue with a physics study

Continue with the tutorials for a
[PXP revival](../tutorials/pxp-z2-revival.md),
[antiferromagnetic chain](../tutorials/antiferromagnetic-chain.md),
or [gate-level GHZ preparation](../tutorials/atom-array-ghz8.md).
For model schemas, units, channel limits, and noise support, use the
[neutral-atom emulator API](../api/atom-emulators.md).
