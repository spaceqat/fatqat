# Emulate a superconducting system

A transmon is a common type of superconducting qubit made from a circuit with
a Josephson junction shunted by a capacitor. Microwave pulses control the
qubit, and a single-qubit gate can take only a few tens of nanoseconds.

Small variations in fabrication mean that even transmons made to the same
design can have different transition frequencies. To produce the intended
rotation on each qubit, measure its transition frequency and tune the pulse
amplitude and phase. On a chip with many qubits, calibration also involves
checking how a pulse affects neighbouring qubits.

A transmon is a nonlinear oscillator with multiple energy levels. Its two
lowest levels, $\lvert 0\rangle$ and $\lvert 1\rangle$, encode the qubit. A
control pulse can also excite higher levels such as $\lvert 2\rangle$. This is
leakage out of the qubit's computational space, a source of gate error that
quantum engineers need to control.

![Three energy levels in a cosine well. Two arrows show that the gap from level 0 to level 1 is larger than the gap from level 1 to level 2.](../assets/generated/guide/transmon-cosine-well.svg)

*In this example with $E_J/E_C=50$, the $0\to1$ transition requires more energy
than the $1\to2$ transition. The difference in transition frequencies helps a
pulse address the qubit transition.*

Pulse emulation lets us study how a choice of pulse affects leakage.
[`TransmonEmulator`][fatqat.emulator.TransmonEmulator] models the lowest three
levels. Following [Write and run pulse programs](hamiltonian-emulation.md),
we use it to compare a reference gate pulse with a waveform you specify
directly on the same transmon.

## Load a single-transmon model

Load the built-in `transmon.single` model. It gives both examples the same
transmon, `q0`, and the same physical parameters.

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model_document = fq.emulator.load_model_document("transmon.single")
>>> model_document["parameters"]["subsystems"]["q0"]["frequency"]
5.1
>>> model = fq.emulator.TransmonModel.from_document(model_document)
>>> model.subsystem_ids
('q0',)
>>> backend = fq.emulator.TransmonEmulator(model, method="density_matrix")
```

Program qubits bind to those subsystem IDs in declaration order unless a
[`ResourceLayout`][fatqat.ResourceLayout] says otherwise. Every model transmon remains
in the physical state even when the Program addresses only one of them.

Retain `model_document` when you need persisted frequencies, anharmonicities,
model identity, or coupling topology. The runtime model provides the execution
capabilities; use the document to look up the stored model data.

## Rotate a qubit with a calibrated pulse { #run-the-rotation-as-a-calibrated-gate }

Add an `RX` gate with angle $\pi/2$ to the Program. The emulator looks up
the reference pulse for that rotation and applies it to the transmon.

```pycon
>>> rotation = fq.Program(1)
>>> rotation.add(ops.RX(np.pi / 2), 0)
>>> calibrated_rho = backend.run(rotation).result().get_density_matrix()
>>> calibrated_rho.shape
(3, 3)
```

The returned density matrix covers all three levels of the transmon. Read its
diagonal to get the probabilities of measuring each level.

```pycon
>>> calibrated_populations = np.real(calibrated_rho.diag())
>>> np.allclose(calibrated_populations.sum(), 1.0)
True
>>> np.round(calibrated_populations[:2], 3)
array([0.5, 0.5])
>>> bool(calibrated_populations[2] < 1e-6)
True
```

After the rotation, $\lvert 0\rangle$ and $\lvert 1\rangle$ each have about
half the population. `calibrated_populations[2]` is the fraction left in
$\lvert 2\rangle$ at the end of the pulse, here less than 0.0001%.

## Drive the transmon directly

Now drive the same transmon with a [`PulseOperation`][fatqat.operations.PulseOperation].
The example uses a 10 ns Hann pulse with an area of $\pi/2$, the target angle
of the reference rotation.

```pycon
>>> duration = 10.0
>>> times = np.linspace(0.0, duration, 41)
>>> rabi_rate = (np.pi / duration) * np.sin(np.pi * times / duration) ** 2
>>> drive = fq.emulator.SampledWaveform(
...     tuple(times),
...     tuple(rabi_rate),
... )
>>> control = fq.emulator.PulseControl(model.control.drive("q0"), drive)
>>> direct = fq.Program(1)
>>> direct.add(ops.PulseOperation(duration, (control,)))
>>> direct_rho = backend.run(direct).result().get_density_matrix()
>>> direct_populations = np.real(direct_rho.diag())
>>> np.round(direct_populations[:2], 3)
array([0.509, 0.491])
>>> f"{100 * direct_populations[2]:.2f}%"
'0.07%'
```

The samples give the Rabi rate $\Omega(t)$ in rad/ns. Here they are real and
drive the X component. An imaginary component would drive Y. The Hann envelope
starts and ends at zero, and the emulator interpolates between the samples.

![The 10 ns Hann pulse rises from zero to its peak at 5 ns and then returns to zero. Dots mark the waveform samples.](../assets/generated/guide/transmon-hann-pulse.svg)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline

    duration = 10.0
    times = np.linspace(0.0, duration, 41)
    rabi_rate = (np.pi / duration) * np.sin(np.pi * times / duration) ** 2
    dense_times = np.linspace(0.0, duration, 401)
    waveform = make_interp_spline(times, rabi_rate)

    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
    ax.plot(dense_times, waveform(dense_times), color="C0")
    ax.scatter(times, rabi_rate, color="C0", s=11, label="Samples")
    ax.set(xlabel="Time (ns)", ylabel="Rabi rate (rad/ns)", xlim=(0, duration))
    ax.legend()
    ```

## Compare the final populations

The direct pulse leaves about 50.9% of the population in $\lvert 0\rangle$,
49.1% in $\lvert 1\rangle$, and 0.07% in $\lvert 2\rangle$. The reference pulse
also divides the population nearly evenly between the qubit states, with
less than 0.0001% left in $\lvert 2\rangle$. The 10 ns Hann pulse and the 20 ns
reference `RX` pulse differ in shape, duration, and DRAG correction. The
plot compares the final populations for these two recipes.

![Both recipes split the population nearly evenly between levels 0 and 1. The magnified panel shows the small population left in level 2 after direct drive.](../assets/generated/guide/transmon-populations.svg)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops

    model = fq.emulator.TransmonModel.from_document(
        fq.emulator.load_model_document("transmon.single")
    )
    backend = fq.emulator.TransmonEmulator(model, method="density_matrix")

    calibrated = fq.Program(1)
    calibrated.add(ops.RX(np.pi / 2), 0)
    calibrated_rho = backend.run(calibrated).result().get_density_matrix()

    duration = 10.0
    times = np.linspace(0.0, duration, 41)
    rabi_rate = (np.pi / duration) * np.sin(np.pi * times / duration) ** 2
    waveform = fq.emulator.SampledWaveform(
        tuple(times),
        tuple(rabi_rate),
    )
    control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
    direct = fq.Program(1)
    direct.add(ops.PulseOperation(duration, (control,)))
    direct_rho = backend.run(direct).result().get_density_matrix()

    calibrated_populations = np.real(calibrated_rho.diag())
    direct_populations = np.real(direct_rho.diag())
    assert np.allclose(calibrated_populations.sum(), 1.0)
    assert np.allclose(direct_populations.sum(), 1.0)

    levels = np.arange(2)
    width = 0.36
    fig, (population_ax, leakage_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.8),
        gridspec_kw={"width_ratios": (2.5, 1.0)},
    )
    population_ax.bar(
        levels - width / 2,
        calibrated_populations[:2],
        width,
        label=r"Reference $R_X(\pi/2)$",
    )
    population_ax.bar(
        levels + width / 2,
        direct_populations[:2],
        width,
        label="Direct drive",
    )
    population_ax.set(
        xticks=levels,
        xticklabels=(r"$|0\rangle$", r"$|1\rangle$"),
        ylabel="Population on q0",
        ylim=(0.0, 1.08),
    )
    population_ax.legend()
    leakage_percent = 100 * np.array((calibrated_populations[2], direct_populations[2]))
    leakage_bars = leakage_ax.bar(
        ("Reference", "Direct"),
        leakage_percent,
        color=("C0", "C1"),
    )
    leakage_labels = [f"{p:.2f}%" if p >= 0.005 else f"{p:.2g}%" for p in leakage_percent]
    leakage_ax.bar_label(leakage_bars, labels=leakage_labels, padding=3)
    leakage_ax.set(ylabel=r"$|2\rangle$ population (%)")
    leakage_ax.set_ylim(0.0, max(0.1, 120 * direct_populations[2]))
    leakage_ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    ```

For a complete amplitude scan, parameter selection, and checks on other input
states, follow [Calibrating a quantum gate](../tutorials/pulse-calibration.md).

## Start with a transmon grid

To try a rectangular grid without writing model and calibration documents by
hand, generate a matching pair and use it to create the emulator:

```pycon
>>> model_document, calibration_document = fq.emulator.generate_transmon_grid_documents(
...     shape=(2, 2),
...     frequency_groups_ghz=(5.0, 5.2),
...     seed=0,
... )
>>> grid_model = fq.emulator.TransmonModel.from_document(model_document)
>>> grid_calibration = fq.emulator.TransmonCalibration(
...     calibration_document
... )
>>> grid_gate_map = fq.emulator.default_transmon_gate_implementation_map(
...     model=grid_model,
...     calibration=grid_calibration,
... )
>>> grid_backend = fq.emulator.TransmonEmulator(
...     grid_model,
...     gate_implementation_map=grid_gate_map,
... )
>>> grid_model.subsystem_ids
('q0', 'q1', 'q2', 'q3')
```

This helper is a convenient way to get a simulation running. It places the two
frequency groups in a checkerboard pattern, adds the requested frequency
spread, and supplies fixed analytic gate recipes. It does not numerically
calibrate the pulses or calculate their fidelity or leakage.

For each edge, the helper chooses the transmon with the higher idle frequency
as the detuned endpoint for CZ. This gives the generated grid a consistent
default; it is not a recommendation for a particular device.

??? example "Save and reload the model and calibration"

    The returned documents are ordinary dictionaries. You can edit them directly
    or save them as JSON and load them again later:

    ```python
    import json

    with open("model.json", "w", encoding="utf-8") as stream:
        json.dump(model_document, stream, indent=2)
    with open("calibration.json", "w", encoding="utf-8") as stream:
        json.dump(calibration_document, stream, indent=2)

    with open("model.json", encoding="utf-8") as stream:
        saved_model = fq.emulator.TransmonModel.from_document(json.load(stream))
    with open("calibration.json", encoding="utf-8") as stream:
        saved_calibration = fq.emulator.TransmonCalibration(json.load(stream))
    ```

For `N` transmons, the physical qutrit Hilbert-space dimension is `3**N`.

The frequency warnings cover the generated idle frequencies only. The RWA
model does not simulate laboratory-frequency collisions. If a pulse tunes the
qubit frequencies, check separately that relevant qubits and spectators do not
cross during the ramp, park, or return.

## Reference pulse model

The reference calibration uses analytic recipes to keep pulse generation
simple. The grid generator uses the same pulse shapes and chooses a CZ branch
for each pair of connected transmons. Further pulse optimization can improve
gate fidelity, for example by reducing the small residual controlled phase in
the reference iSWAP.

The emulator keeps three levels per transmon and uses an effective rotating-
frame/RWA Hamiltonian. In the solver, \(\alpha_i\), \(\delta_i\),
\(\Omega_i\), and \(g_{ij}\) are angular rates:

$$
H(t) =
\sum_i \frac{\alpha_i}{2} n_i(n_i-1)
+ \sum_i \delta_i(t)n_i
+ \frac{1}{2}\sum_i \left[\Omega_i(t)a_i^\dagger
                 + \Omega_i^*(t)a_i\right]
+ \sum_{(i,j)\in E} g_{ij}(t)
  \left(a_i^\dagger a_j+a_i a_j^\dagger\right),
\qquad n_i=a_i^\dagger a_i.
$$

Here \(a_i\) is the three-level lowering operator. The signed anharmonicity
\(\alpha_i\) is the only coherent drift. Drive, detuning, and exchange appear
only while their controls are active. In particular, there is no static
exchange. Exchange is allowed only on a declared model edge; generated
rectangular grids connect horizontal and vertical nearest neighbours.

??? example "iSWAP pulse recipe"

    The reference iSwap is a 40 ns Hann exchange pulse. For duration \(T\),

    $$
    g_{ij}(t) = -\frac{\pi}{T}\sin^2\!\left(\frac{\pi t}{T}\right),
    \qquad 0\leq t\leq T,
    $$

    so \(\int_0^T g_{ij}(t)\,dt=-\pi/2\). A frame swap after the pulse gives
    the public \(+i\) convention. The full qutrit exchange operator also couples
    \(\lvert11\rangle\) to \(\lvert20\rangle\) and \(\lvert02\rangle\), so
    conditional-phase error and residual leakage remain possible.

??? example "CZ pulse recipe"

    For CZ, label the selected endpoint \(i\). In a basis that lists it first, the
    chosen branch is
    \(\lvert11\rangle\leftrightarrow\lvert20\rangle\). The recipe parks that
    endpoint with \(\delta_i(t)=-\alpha_i s(t)\). For total duration \(T\) and
    ramp time \(t_r\),

    $$
    s(t)=
    \begin{cases}
    \frac{1-\cos(\pi t/t_r)}{2}, & 0\leq t<t_r,\\
    1, & t_r\leq t\leq T-t_r,\\
    \frac{1-\cos(\pi(T-t)/t_r)}{2}, & T-t_r<t\leq T.
    \end{cases}
    $$

    Exchange is active only during the parked interval \(T_p=T-2t_r\):

    $$
    g_{ij}(t) = \frac{\sqrt{2}\pi}{T_p}
    \sin^2\!\left(\frac{\pi(t-t_r)}{T_p}\right),
    \qquad t_r\leq t\leq T-t_r.
    $$

    The nominal resonant branch closes one cycle, and a frame update removes the
    local phase accumulated from detuning. The defaults are 60 ns total with 3 ns
    ramps.

    Because exchange is zero during the ramps, this recipe does not model
    Landau–Zener leakage from crossing an avoided crossing with finite coupling.
    It also omits a tunable coupler, residual exchange, waveform distortion, and
    non-RWA terms. Final leakage can therefore be tiny even when the transient
    \(\lvert20\rangle\) population is large. Changing the duration preserves the
    nominal exchange area; it does not recalibrate phases or fidelity.

## Know when more physics changes the answer

- **Coupling:** two-transmon gates are available only on coupling edges in the
  model, and an unaddressed neighbour is still part of the Hamiltonian.
- **Frames and timing:** calibrated frame changes and the placement of later
  controls can change phases even when computational populations look alike.
  Returned states include terminal virtual-frame changes.
- **Continuous noise:** rate- or time-form Lindblad declarations act over
  elapsed physical time, rather than once at a circuit-operation boundary.

Use the calibrated path to study the supplied gate recipe. Use direct controls
when the waveform itself is the experiment; the two forms can also coexist in
one Program. The [transmon emulator API](../api/pulse-emulator.md) lists the
supported units, gates, noise forms, and simulation methods.
