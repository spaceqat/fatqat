---
title: "LOSimulator"
---

# LOSimulator

[`LOSimulator`][fatqat.simulator.LOSimulator] runs programs with photons in
optical modes. Use it to compute the output photon counts or amplitudes of a
linear-optical circuit.

## Configure an optical run { #choose-an-optical-simulation-method }

Use the default `fixed_photon_state` method when the program uses only a
[`ModeRegister`][fatqat.ModeRegister] or several such registers, starts in a
[`BasisState`][fatqat.simulator.BasisState] or the vacuum, and preserves the total
photon number. It stores amplitudes only for the arrangements possible at that
photon number.

Use `statevector` or `density_matrix` for a dense initial state, for reset or
noise channels, or for a program that mixes modes with qubits. The simulator also
provides the [operator methods](../simulator.md#methods).

When passing a `noise` model, use device labels to select optical modes.

### Set the photon cutoff { #set-the-photon-cutoff }

The `cutoff` sets the maximum number of photons per mode for one run. Pass it in
`execution_config` to set the range from `0` through `cutoff` for each mode and
each entry in a [`CountRegister`][fatqat.CountRegister].

When you supply a `BasisState`, the simulator can infer the cutoff from the total
number of photons, with a minimum of one. It can also infer the cutoff from a
compatible dense initial array. For a vacuum run with no `initial_state`, set
`cutoff` explicitly.

## Read optical results { #read-optical-results }

For the fixed-photon method,
[`Result.get_fixed_photon_state()`][fatqat.Result.get_fixed_photon_state] gives
one amplitude per arrangement of photons. Use
[`fixed_photon_index()`][fatqat.simulator.fixed_photon_index] to find an
arrangement's position, and
[`fixed_photon_occupation()`][fatqat.simulator.fixed_photon_occupation] to find
the arrangement at a given position.

Pass the photon counts in the mode order recorded in `metadata["state_axes"]`.

```pycon
>>> from fatqat.simulator import fixed_photon_index, fixed_photon_occupation
>>> fixed_photon_index((1, 1, 0))
4
>>> fixed_photon_occupation(4, num_modes=3, num_photons=2)
(1, 1, 0)
```

## API reference { #api }

<a id="fatqat.simulator.LOSimulator.method"></a>

<a id="fatqat.simulator.LOSimulator.run"></a>

<a id="fatqat.simulator.LOSimulator.run_sweep"></a>

<a id="fatqat.simulator.LOSimulator.validate_noise_model"></a>

Use [`Simulator.run()`][fatqat.simulator.Simulator.run] for a single program and
[`Simulator.run_sweep()`][fatqat.simulator.Simulator.run_sweep] for a parameter
sweep. The entries below cover the simulator-specific interfaces.

::: fatqat.simulator.LOSimulator
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.simulator.fixed_photon_index

::: fatqat.simulator.fixed_photon_occupation
