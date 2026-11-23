# Emulate neutral atoms with laser pulses

Optical tweezers hold individual neutral atoms at chosen positions. A laser
can drive an atom from its ground state $\lvert g\rangle$ to an excited Rydberg
state $\lvert r\rangle$. When two nearby atoms are both in $\lvert r\rangle$,
their interaction shifts the energy of the pair. A pulse tuned to excite one
atom then has a much harder time exciting both. This suppression is called
Rydberg blockade.

With [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator], you can study
that effect using the [pulse workflow](hamiltonian-emulation.md). The example
below changes only the distance between two atoms and checks how often both
are excited at the end of the same pulse. For loading, moving, or pairing
atoms in a gate-level Program, see [Neutral-atom array simulation](atom-array-simulation.md).

## Drive two atoms with the same pulse { #drive-the-array-with-the-two-level-model }

The example starts with both atoms in $\lvert g\rangle$. Write one resonant
drive pulse that acts on both atoms for 1 µs, with a peak Rabi rate of 3 rad/µs.
The two-qubit Program provides one two-level resource per atom. The emulator
gets the positions separately from an
[`AtomArrangement`][fatqat.emulator.AtomArrangement].

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.Atom2LevelModel.from_document(
...     fq.emulator.load_model_document("atom2level.reference")
... )
>>> duration = 1.0
>>> times = (0.0, 0.5, duration)
>>> drive = fq.emulator.SampledWaveform(times, (0.0, 3.0, 0.0))
>>> drive_control = fq.emulator.PulseControl(model.control.drive(), drive)
>>> program = fq.Program(2)
>>> program.add(ops.PulseOperation(duration, (drive_control,)))
```

Run that Program at 4 µm and 12 µm. The statevector lists amplitudes in the
order $\lvert gg\rangle$, $\lvert gr\rangle$, $\lvert rg\rangle$, and
$\lvert rr\rangle$. Square their magnitudes to get the probabilities, and
add the middle two to find the probability of exactly one excitation.

```pycon
>>> print("Distance   0 excited   1 excited   2 excited")
Distance   0 excited   1 excited   2 excited
>>> for distance in (4.0, 12.0):
...     arrangement = fq.emulator.AtomArrangement.chain(2, spacing=distance)
...     emulator = fq.emulator.Atom2LevelEmulator(model, arrangement=arrangement)
...     state = emulator.run(program).result().get_statevector()
...     probabilities = np.abs(state.full().ravel()) ** 2
...     zero = probabilities[0]
...     one = probabilities[1:3].sum()
...     two = probabilities[3]
...     print(f"{distance:>2g} µm       {zero:>7.3%}     {one:>7.3%}     {two:>7.3%}")
 4 µm        2.502%     97.496%      0.002%
12 µm        8.521%     41.347%     50.132%
```

At 4 µm, the pulse almost always excites exactly one atom, and the chance of
exciting both is about 0.002%. At 12 µm, the chance of exciting both rises to
about 50%. In this model, the interaction between two Rydberg excitations
scales as $C_6/R^6$, where $R$ is their separation. The larger separation
weakens the interaction, so the same pulse can excite both atoms much more
readily.

## Describe other arrangements { #describe-the-sites-once }

Each arrangement fixes the positions used by the emulator. By default, it
maps Program resources to those positions in declaration order. Use a
[`ResourceLayout`][fatqat.ResourceLayout] to supply a different mapping.

To specify positions yourself, pass $(x, y)$ coordinates for two dimensions
or $(x, y, z)$ coordinates for three. Use the same number of components for
every site. Distances are in micrometres.

```pycon
>>> planar = fq.emulator.AtomArrangement.from_coordinates([(0, 0), (3, 4)])
>>> planar.spatial_dimension
2
>>> spatial = fq.emulator.AtomArrangement.from_coordinates([(0, 0, 0), (3, 4, 5)])
>>> spatial.spatial_dimension
3
```

For a rectangular array, you can set separate spacings along $x$ and $y$. To
describe several regions in one arrangement, combine them in the desired site
order and use labels to look up their indices.

```pycon
>>> storage = fq.emulator.AtomArrangement.rectangular(
...     rows=2, cols=2, spacing=(4.0, 6.0), label="storage"
... )
>>> interaction = fq.emulator.AtomArrangement.from_coordinates(
...     [(12, 0), (12, 6)], label="interaction"
... )
>>> combined = fq.emulator.AtomArrangement.combine(storage, interaction)
>>> combined.group("storage")
(0, 1, 2, 3)
>>> combined.group("interaction")
(4, 5)
```

See the [arrangement API](../api/atom-emulators.md#arrangements-and-program-resources)
for coordinate and group rules.

## Add detuning to the drive

Detuning sets how far the drive frequency is from the atomic transition.
Use a second control alongside the drive to vary it during the pulse. Both
channels act on every atom and use rad/µs.

```python
detuning = fq.emulator.SampledWaveform(times, (0.0, 0.1, 0.0))
detuning_control = fq.emulator.PulseControl(model.control.detuning(), detuning)
detuned_program = fq.Program(2)
detuned_program.add(
    ops.PulseOperation(duration, (drive_control, detuning_control))
)
```

The emulator includes interactions between every pair of atoms by default.
Use `execution_config={"interaction_cutoff": distance}` on a run to truncate
the Hamiltonian by dropping pairs farther apart than that distance.

For units, channel limits, noise, and measurement support, see the
[neutral-atom emulator API](../api/atom-emulators.md).
