# Simulate linear optics

Linear optics computes with photons travelling through optical modes, such as
the two output ports of a beam splitter. It differs from qubit circuits in
three ways:

- **A mode holds any number of photons.** Its level `n` is a photon count, not
  one of two states.
- **Photons are moved, never created.** Beam splitters and phase shifters
  redistribute photons among modes; the total number that entered stays the
  same.
- **Photons are identical.** When two photons could have taken different paths
  to the same outcome, those paths interfere, and some outcomes disappear.

This guide shows all three with the Hong–Ou–Mandel experiment, run on
[`LOSimulator`][fatqat.simulator.LOSimulator].

## One photon on a beam splitter

A [`ModeRegister`][fatqat.ModeRegister] holds the modes. Measuring a mode
records its photon count in a [`CountRegister`][fatqat.CountRegister]. Build a
balanced (50:50) [`BeamSplitter`][fatqat.operations.BeamSplitter] between two
modes and count the photons in each output:

```pycon
>>> import math
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> from fatqat.simulator import BasisState, LOSimulator
>>> modes = fq.ModeRegister(2)
>>> counts = fq.CountRegister(2)
>>> splitter = fq.Program([modes], [counts])
>>> splitter.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
>>> splitter.measure((modes[0], modes[1]), (counts[0], counts[1]))
```

Photons enter through the run's `initial_state`: a
[`BasisState`][fatqat.simulator.BasisState] gives the photon count of each
mode, and unnamed modes start empty. Send one photon into mode 0:

```pycon
>>> simulator = LOSimulator(runtime="numpy")
>>> one_photon = BasisState({modes[0]: 1})
>>> simulator.run(
...     splitter, shots=1000, initial_state=one_photon,
...     simulation_config={"seed": 7},
... ).result().get_counts()
{'01': 502, '10': 498}
```

Each key lists the counts in register order: `"10"` is the photon in mode 0,
`"01"` in mode 1. The photon leaves through either port with probability ½,
like a fair coin.

## Two photons: Hong–Ou–Mandel interference

Now send one photon into each input. If the photons behaved like two
independent coins, a quarter of the runs would put both in mode 0, a quarter
both in mode 1, and half would put one in each mode, `"11"`:

```pycon
>>> two_photons = BasisState({modes[0]: 1, modes[1]: 1})
>>> simulator.run(
...     splitter, shots=1000, initial_state=two_photons,
...     simulation_config={"seed": 7},
... ).result().get_counts()
{'02': 502, '20': 498}
```

The outcome `"11"` never occurs; the photons always leave together. To end in
`"11"`, both photons must pass straight through or both must be reflected.
Because the photons are identical, these two histories lead to the same final
state, so their amplitudes add, and at a 50:50 splitter they cancel exactly.

![Four beam-splitter diagrams show the histories of two photons entering modes 0 and 1. Both photons passing through and both reflecting each end in outcome 11; their amplitudes cos theta squared and minus sin theta squared add to cos 2 theta, which is zero at 50:50. One photon passing and the other reflecting puts both photons in mode 0 or both in mode 1.](../assets/generated/guide/hong-ou-mandel-histories.svg)

This is Hong–Ou–Mandel interference, and it has no counterpart for classical
particles.

## Change the reflectivity

The cancellation is exact only for a balanced splitter. `BeamSplitter(theta)`
reflects a photon with probability `R = sin(theta)**2`. The two histories that
end in `"11"` have amplitudes `cos(theta)**2` and `-sin(theta)**2`, so the
coincidence probability is `(1 - 2R)**2`. For distinguishable particles the
probabilities would add instead, giving `(1 - R)**2 + R**2`.

An [`Estimator`][fatqat.Estimator] computes the exact probability of `"11"` as
the expectation value of [`Projector(1)`][fatqat.observable.Projector] on both
modes. Sweep the angle to compare the two:

```pycon
>>> from fatqat.observable import Projector
>>> theta = fq.Parameter("theta")
>>> mixer = fq.Program([modes])
>>> mixer.add(ops.BeamSplitter(theta), (modes[0], modes[1]))
>>> coincidence = fq.Observable.from_sparse(
...     [([Projector(1), Projector(1)], (0, 1), 1.0)]
... )
>>> angles = [0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2]
>>> rows = fq.Estimator(simulator).run_sweep(
...     mixer, coincidence, {theta: angles}, initial_state=two_photons
... ).result()
>>> for angle, row in zip(angles, rows):
...     R = math.sin(angle) ** 2
...     print(f"R={R:.3f}  photons: {float(row.get_expectation()):.3f}  "
...           f"distinguishable: {(1 - R) ** 2 + R ** 2:.3f}")
R=0.000  photons: 1.000  distinguishable: 1.000
R=0.146  photons: 0.500  distinguishable: 0.750
R=0.500  photons: 0.000  distinguishable: 0.500
R=0.854  photons: 0.500  distinguishable: 0.750
R=1.000  photons: 1.000  distinguishable: 1.000
```

A finer sweep traces the full curves:

![Coincidence probability against reflectivity. For distinguishable particles the curve dips to one half at reflectivity one half; for identical photons it dips to zero there. Both curves equal one at reflectivity zero and one.](../assets/generated/guide/hong-ou-mandel-reflectivity.svg)

??? example "Reproduce this figure"

    ```python
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops
    from fatqat.observable import Projector
    from fatqat.simulator import BasisState, LOSimulator

    modes = fq.ModeRegister(2)
    theta = fq.Parameter("theta")
    mixer = fq.Program([modes])
    mixer.add(ops.BeamSplitter(theta), (modes[0], modes[1]))
    coincidence = fq.Observable.from_sparse(
        [([Projector(1), Projector(1)], (0, 1), 1.0)]
    )
    angles = np.linspace(0.0, math.pi / 2, 61)
    rows = fq.Estimator(LOSimulator(runtime="numpy")).run_sweep(
        mixer,
        coincidence,
        {theta: angles},
        initial_state=BasisState({modes[0]: 1, modes[1]: 1}),
    ).result()

    reflectivity = np.sin(angles) ** 2
    photons = [float(row.get_expectation()) for row in rows]
    distinguishable = (1 - reflectivity) ** 2 + reflectivity**2

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(reflectivity, distinguishable, "--", color="0.45",
              label="Distinguishable particles")
    axis.plot(reflectivity, photons, color="C0", label="Identical photons")
    axis.set(xlabel="Reflectivity R", ylabel='Probability of "11"',
             xlim=(0, 1), ylim=(0, 1.05))
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plt.show()
    ```

At `R = 0` and `R = 1` only one history is possible, so the two cases agree.
In between, interference suppresses coincidences, and at `R = 0.5` it removes
them completely. In the laboratory, the Hong–Ou–Mandel dip is usually traced
at a fixed 50:50 splitter by delaying one photon, which makes the photons
partly distinguishable and brings the coincidences back.

## Simulate larger interferometers

A statevector stores every combination of photon counts, from `0` to the
cutoff in each mode. Twelve modes with four photons would need `5**12`, about
244 million amplitudes or 3.9 GB, although almost all of them stay zero: beam
splitters never change the total photon number.

`LOSimulator`'s default method, `"fixed_photon_state"`, stores only the
arrangements of the photons you sent in, 1365 for four photons in twelve
modes. All runs in this guide use it. This interferometer mixes neighboring
modes in twelve layers:

```pycon
>>> big_modes = fq.ModeRegister(12)
>>> mesh = fq.Program([big_modes])
>>> for layer in range(12):
...     for i in range(layer % 2, 11, 2):
...         mesh.add(ops.BeamSplitter(math.pi / 4), (big_modes[i], big_modes[i + 1]))
...         mesh.add(ops.PhaseShifter(0.1 * (i + layer)), big_modes[i])
>>> four_photons = BasisState({big_modes[i]: 1 for i in range(4)})
>>> state = simulator.run(
...     mesh, initial_state=four_photons
... ).result().get_fixed_photon_state()
>>> len(state)
1365
```

Each amplitude belongs to one arrangement of the four photons.
[`fixed_photon_index`][fatqat.simulator.fixed_photon_index] finds the
amplitude of an arrangement, given as photon counts in mode order. The
probability that the photons leave in the four modes they entered is

```pycon
>>> from fatqat.simulator import fixed_photon_index
>>> back_where_they_started = (1, 1, 1, 1) + (0,) * 8
>>> round(float(abs(state[fixed_photon_index(back_where_they_started)]) ** 2), 4)
0.0056
```

The method needs a `BasisState` start and modes only, and it rejects noise.
For qubits alongside modes, dense initial states, or noise, choose
`LOSimulator("statevector")` or `LOSimulator("density_matrix")`; see
[`LOSimulator`][fatqat.simulator.LOSimulator] for the full rules.
