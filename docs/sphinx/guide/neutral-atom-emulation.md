# Choose and run a neutral-atom workflow

FatQat offers three neutral-atom execution levels. Choose the least detailed
one that still contains the effect you want to study:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Gate-level array
{py:class}`~fatqat.simulator.AtomArraySimulator` adds occupancy and a changing
pairing graph to finite qubit gates. Use it for fast loading, loss, and
connectivity checks.
:::

:::{grid-item-card} Three physical levels
{py:class}`~fatqat.emulator.Atom3LevelEmulator` follows $\lvert 0\rangle$,
$\lvert 1\rangle$, and $\lvert r\rangle$. Use it for calibrated gates,
selected-site control, and Rydberg leakage.
:::

:::{grid-item-card} Two physical levels
{py:class}`~fatqat.emulator.Atom2LevelEmulator` follows $\lvert g\rangle$ and
$\lvert r\rangle$. Use it for global drive, detuning, and many-body Rydberg
dynamics.
:::

::::

The atom-array simulator is the profile introduced in
{doc}`Hardware-profile simulation <hardware-profile-simulation>`; it does not
integrate a Hamiltonian. The two emulators below use the pulse workflow from
{doc}`Hamiltonian emulation <hamiltonian-emulation>`.

## Describe the sites once

Both physical emulators use a fixed set of sites described by an
{py:class}`~fatqat.emulator.AtomArrangement`. Program resources map to those
sites in declaration order by default:

```{doctest}
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

Use a {py:class}`~fatqat.ResourceLayout` only when you need to override the
default declaration-order mapping.

## Address one site with the three-level model

Load the three-level reference model and run an ordinary calibrated rotation:

```{doctest}
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

```{doctest}
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

```{doctest}
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
{doc}`PXP revival <../tutorials/plot_pxp_z2_revival>`,
{doc}`antiferromagnetic chain <../tutorials/plot_atom2level_antiferromagnetic_chain>`,
or {doc}`gate-level GHZ preparation <../tutorials/plot_atom_array_ghz8>`.
For model schemas, units, channel limits, and noise support, use the
{doc}`neutral-atom emulator API <../api/atom-emulators>`.
