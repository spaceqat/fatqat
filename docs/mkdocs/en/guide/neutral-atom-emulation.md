# Choose and run a neutral-atom workflow

FatQat offers three neutral-atom execution levels. Choose the least detailed
one that still contains the effect you want to study:

<div class="grid cards" markdown>

-   **Gate-level array**

    [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] adds occupancy and a changing
    pairing graph to finite qubit gates. Use it for fast loading, loss, and
    connectivity checks.

-   **Three physical levels**

    [`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator] follows $\lvert 0\rangle$,
    $\lvert 1\rangle$, and $\lvert r\rangle$. Use it for calibrated gates,
    selected-site control, and Rydberg leakage.

-   **Two physical levels**

    [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] follows $\lvert g\rangle$ and
    $\lvert r\rangle$. Use it for global drive, detuning, and many-body Rydberg
    dynamics.

</div>

The atom-array simulator is the profile introduced in
[Hardware-profile simulation](hardware-profile-simulation.md); it does not
integrate a Hamiltonian. The two emulators below use the pulse workflow from
[Hamiltonian emulation](hamiltonian-emulation.md).

## Describe the sites once

Both physical emulators use a fixed set of sites described by an
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
exactly one dimension-two resource per site. In the three-level emulator,
those logical qubit resources are embedded into physical qutrits; that does
not turn them into logical qutrit resources.

Use a [`ResourceLayout`][fatqat.ResourceLayout] only when you need to override the
default declaration-order mapping.

## Address one site with the three-level model

Load the three-level reference model and run an ordinary calibrated rotation:

```pycon
>>> atom3_model = fq.emulator.Atom3LevelModel.from_document(
...     fq.emulator.load_model_document("atom3level.reference")
... )
>>> atom3 = fq.emulator.Atom3LevelEmulator(
...     atom3_model,
...     arrangement=arrangement,
... )
>>> calibrated = fq.Program(arrangement.num_sites)
>>> calibrated.add(ops.RX(np.pi / 2), 0)
>>> atom3_rho = atom3.run(calibrated).result().get_density_matrix()
>>> atom3_rho.shape
(9, 9)
```

The `(9, 9)` result retains physical `|r>` population for both sites. The
built-in map also realizes `RY`, `RZ`, and `CZ`; use it when the packaged
calibration is the experiment's starting point.

For selected-site control, address a Raman or Rydberg channel by site. This
direct block first drives the Raman transition on site `0`, then starts a
Rydberg waveform halfway through the operation:

```pycon
>>> shaped = fq.emulator.SampledWaveform(
...     (0.0, 0.25, 0.5),
...     (0.0, 4.0, 0.0),
... )
>>> controls = (
...     fq.emulator.PulseControl(atom3_model.control.raman(0), shaped),
...     fq.emulator.PulseControl(
...         atom3_model.control.rydberg(0),
...         shaped,
...         start_offset=0.5,
...     ),
... )
>>> selected_site = fq.Program(arrangement.num_sites)
>>> selected_site.add(ops.PulseOperation(1.0, controls))
>>> selected_rho = atom3.run(selected_site).result().get_density_matrix()
>>> physical = np.real(np.diag(selected_rho)).reshape((3, 3), order="F")
>>> round(float(physical[2].sum()), 3)
0.146
```

Here the last value is site `0`'s physical Rydberg population. Geometry also
contributes an interaction whenever more than one site has Rydberg
population, including sites not named by the same control block.

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
pair interacts; `interaction_cutoff` can remove terms beyond a distance, but
it is a Hamiltonian truncation rather than a blockade radius.

For this short pulse, about 5.4% of the final probability lies outside the
all-ground state. That number is a physical consequence of the waveform,
detuning, spacing, interaction model, and duration—not a gate label.

An ideal, unmeasured run returns the complete two-level statevector. With
supported Lindblad noise it can instead return a density matrix, and terminal
measurement returns counts. Mid-circuit measurement, reset, conditions, and
per-site direct controls are not part of this two-level workflow.

## Continue with a physics study

Continue with the tutorials for a
[PXP revival](../tutorials/pxp-z2-revival.md),
[antiferromagnetic chain](../tutorials/antiferromagnetic-chain.md),
or [gate-level GHZ preparation](../tutorials/atom-array-ghz8.md).
For model schemas, units, channel limits, and noise support, use the
[neutral-atom emulator API](../api/atom-emulators.md).
