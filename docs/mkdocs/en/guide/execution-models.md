# Choose how much physics to model

Choose a backend according to the question you want to answer. A circuit
simulation can test an algorithm, a hardware profile can check device rules,
and a physical emulator can resolve what happens while a pulse is applied.

| If you want to know… | Use… | FatQat models… |
|---|---|---|
| whether a circuit produces the intended result, or how noise channels change it | the general simulator | circuit operations on qubits or qudits, with optional noise channels |
| whether a Program uses a device's supported operations and resources | a hardware-profile simulator | circuit evolution with native-operation, placement, connectivity, capacity, and occupancy checks, plus optional noise |
| how pulse shape, duration, or drift affects the state during execution | a physical emulator | time-dependent evolution of the model's physical levels under controls, coupling, and compatible continuous noise |

These are choices for different questions. You can begin with any of them.
In particular, studying noise does not by itself require an emulator: both
circuit simulators and hardware profiles support noise channels. Use physical
emulation when you need to resolve dynamics over elapsed time, such as
excitation during a pulse or relaxation during an idle interval.

## Compare a single-qubit rotation

An ideal `RX(pi/2)` rotation of a qubit initially in `|0>` gives equal
probabilities for `0` and `1`. The three backends let you examine this operation
under different assumptions:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> program = fq.Program(1)
>>> program.add(ops.RX(np.pi / 2), 0)
```

=== "General simulator"

    The general [`Simulator`][fatqat.simulator.Simulator] applies the circuit
    operation without assigning the qubit to a particular device. With no noise
    model, this run gives the ideal probabilities:

    ```pycon
    >>> general = fq.simulator.Simulator(method="statevector", runtime="numpy")
    >>> general_result = general.run(
    ...     program,
    ...     shots=0,
    ...     result_config={"final_state": True},
    ... ).result()
    >>> np.round(np.abs(general_result.get_statevector()) ** 2, 3)
    array([0.5, 0.5])
    ```

=== "Hardware profile"

    A hardware-profile simulator also evolves circuit operations, while checking
    that they belong to its native set and obey its resource rules. The
    superconducting profile expresses this rotation as `SX`, which has the same
    effect as `RX(pi/2)` up to a global phase:

    ```pycon
    >>> profile_program = fq.Program(1)
    >>> profile_program.add(ops.SX, 0)
    >>> profile = fq.simulator.SCQubitSimulator(
    ...     num_qubits=1,
    ...     couplings=(),
    ...     runtime="numpy",
    ... )
    >>> profile_result = profile.run(
    ...     profile_program,
    ...     shots=0,
    ...     result_config={"final_state": True},
    ... ).result()
    >>> np.round(np.abs(profile_result.get_statevector()) ** 2, 3)
    array([0.5, 0.5])
    ```

    The probabilities match the general simulator's. This run also checks that
    `SX` is supported on the selected qubit; it does not model the pulse used to
    implement it.

    Calling `profile.run(program)` with the original `RX(pi/2)` operation
    would raise [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]
    because `RX` is not native to this profile.

=== "Physical emulator"

    The [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] turns `RX` into a
    pulse using its packaged reference calibration, then integrates the physical
    state over the pulse's duration:

    ```pycon
    >>> model = fq.emulator.TransmonModel.from_document(
    ...     fq.emulator.load_model_document("transmon.reference")
    ... )
    >>> emulator = fq.emulator.TransmonEmulator(model)
    >>> physical_result = emulator.run(program, shots=0).result()
    >>> physical_state = physical_result.get_statevector().full().ravel()
    >>> joint_populations = (np.abs(physical_state) ** 2).reshape(3, 3)
    >>> q0_populations = joint_populations.sum(axis=1)
    >>> np.round(q0_populations, 3)
    array([0.5, 0.5, 0. ])
    ```

    The Program declares one qubit with two basis states. The reference model
    contains two physical transmons with three levels each, so the returned
    state has nine amplitudes. Reshaping it gives populations indexed by the
    levels of `q0` and `q1`; summing over `q1` leaves the three populations on
    `q0` shown above.

    Level `|2>` represents leakage out of the qubit's `|0>`, `|1>` subspace.
    Retaining that physical level does not make the Program a qutrit program.
    The second transmon also remains in the state even though this Program does
    not address it. For this reference pulse, the final populations are close
    to the ideal values and the leakage rounds to zero at this precision. The
    packaged calibration supplies a reproducible simulation example, not a
    current hardware characterization.

## See what happens during a pulse

Equal final probabilities do not show how the state changes during a drive.
For a simple time-resolved example, apply a constant drive to `q0` and stop it
at successively later times. This is a direct pulse, separate from the shaped
reference pulse used for `RX` above. Its fixed Rabi rate is chosen so that an
ideal two-level system would reach equal populations after 10 ns.

![A constant drive transfers population from level zero to level one over 10 nanoseconds; a second panel magnifies the small population in leakage level two.](../assets/generated/guide/execution-models-pulse.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops

    model = fq.emulator.TransmonModel.from_document(
        fq.emulator.load_model_document("transmon.reference")
    )
    emulator = fq.emulator.TransmonEmulator(model)
    times = np.linspace(0.0, 10.0, 101)  # ns
    rabi_rate = np.pi / 20.0  # rad/ns, held fixed for every run
    populations = []

    for stop_time in times:
        pulse_program = fq.Program(1)
        if stop_time > 0:
            waveform = fq.emulator.SampledWaveform(
                (0.0, stop_time), (rabi_rate, rabi_rate)
            )
            control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
            pulse_program.add(ops.PulseOperation(stop_time, (control,)))
        state = emulator.run(pulse_program, shots=0).result().get_statevector()
        state = state.full().ravel()
        populations.append((np.abs(state) ** 2).reshape(3, 3).sum(axis=1))

    populations = np.array(populations)
    assert np.allclose(populations.sum(axis=1), 1.0)

    figure, (population_ax, leakage_ax) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), sharex=True
    )
    for level in (0, 1):
        population_ax.plot(times, populations[:, level], label=f"|{level}>")
    population_ax.set(xlabel="Time (ns)", ylabel="Population on q0", ylim=(0, 1))
    population_ax.legend()
    leakage_ax.plot(times, 100 * populations[:, 2], color="C2")
    leakage_ax.set(xlabel="Time (ns)", ylabel="|2> population (%)")
    for axis in (population_ax, leakage_ax):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    plt.show()
    ```

The left panel shows population moving between the qubit levels. The right
panel uses a smaller scale to reveal population in `|2>`. This leakage arises
from the driven three-level dynamics, even though no noise model is attached.
To compare shaped controls with the calibrated rotation, continue with
[Transmon emulation](transmon-emulation.md#drive-the-transmon-directly).

## Moving between models

You can reuse a Program when the destination backend supports its operations
and resources. Otherwise, express the computation using operations it accepts,
as `SX` does for the superconducting profile above. When program resources
need specific device locations, a [`ResourceLayout`][fatqat.ResourceLayout]
binds them to device labels. [Hardware-profile simulation](hardware-profile-simulation.md)
walks through a placement error and its correction.

For circuit evolution and noise channels, see [Simulation](simulation.md) and
[Ideal and noisy runs](ideal-and-noisy.md). For calibrated gates and direct
controls, see [Hamiltonian emulation](hamiltonian-emulation.md). The
[backend noise-support table](../api/noise/backend-support.md#noise-backend-support)
helps you choose compatible noise declarations when changing models.
