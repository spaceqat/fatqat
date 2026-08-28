# Choose how much physics to model

A backend is more than a place to send a program. It decides what FatQat
means by “run this.” The general simulator follows logical circuit evolution;
a hardware profile adds device rules; an emulator follows a physical model in
time.

The distinction is easiest to see when the `Program` does not change. This
single-qubit rotation is understood by all three examples below:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> program = fq.Program(1)
>>> program.add(ops.RX(np.pi / 2), 0)
```

=== "General simulator"

    The general [`Simulator`][fatqat.simulator.Simulator] applies the logical gate
    operation without assigning the qubit to a particular device:

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

    This is the quickest answer to the algorithmic question: the rotation leaves
    equal probabilities for `0` and `1`.

=== "Hardware profile"

    A hardware-profile simulator still evolves gates at circuit level, but first
    checks that the program is native and physically placeable. `RX` is native to
    the Google-style profile used here:

    ```pycon
    >>> profile = fq.simulator.SCQubitGoogleSimulator(
    ...     grid_size=(1, 1),
    ...     runtime="numpy",
    ... )
    >>> profile_result = profile.run(
    ...     program,
    ...     shots=0,
    ...     result_config={"final_state": True},
    ... ).result()
    >>> np.round(np.abs(profile_result.get_statevector()) ** 2, 3)
    array([0.5, 0.5])
    ```

    The numerical answer matches, but the claim is stronger: this particular
    instruction also belongs to the selected native gate set and fits its
    resource model.

=== "Physical emulator"

    The transmon emulator realizes `RX` through its packaged pulse calibration and
    integrates a three-level physical model:

    ```pycon
    >>> model = fq.emulator.TransmonModel.from_document(
    ...     fq.emulator.load_model_document("transmon.reference")
    ... )
    >>> emulator = fq.emulator.TransmonEmulator(model)
    >>> physical_result = emulator.run(program, shots=0).result()
    >>> physical_state = physical_result.get_density_matrix()
    >>> physical_state.shape
    (9, 9)
    >>> round(float(np.trace(physical_state).real), 12)
    1.0
    ```

    The reference model contains two physical three-level transmons, so its state
    is larger than the two-amplitude logical state—even though the `Program`
    addresses only one qubit. That extra space is where unaddressed hardware and
    leakage live.

## What each level tells you

| If you want to know… | Start with… | FatQat models… |
|---|---|---|
| whether the algorithm produces the intended logical behavior | the general simulator | circuit operations on logical subsystems |
| whether the written program obeys a target's native operations and resource rules | a hardware-profile simulator | circuit evolution plus layout, connectivity, occupancy, and optional profile noise |
| how gates or controls behave as timed physical dynamics | an emulator | levels, drift, coupling, pulses, Hamiltonians, and compatible open-system noise |

Start with the simplest model that contains the effect you need. Add device
rules or continuous-time physics only when those details affect the result.

## Expect different capability checks

All three paths accept a `Program`, return a `Job`, and expose data through a
`Result`, but they do not accept the same instruction set.

For example, the general simulator supports logical qudits and registers with
mixed local dimensions. Current hardware profiles and pulse emulators accept
dimension-two logical resources. A transmon emulator may return a physical
qutrit state because it retains a leakage level; that is different from
declaring a logical qutrit in the Program.

Hardware profiles validate the Program as written; they do not transpile or
route it. When logical and device labels differ, a
[`ResourceLayout`][fatqat.ResourceLayout] makes the binding explicit. [Testing a
hardware profile](hardware-profile-simulation.md) works through a failed
placement and its correction.

Next, [simulate a quantum program](simulation.md) to explore states, sweeps,
and outputs at circuit level, or jump to [Hamiltonian-level
emulation](hamiltonian-emulation.md) when time and controls are already the
focus.
