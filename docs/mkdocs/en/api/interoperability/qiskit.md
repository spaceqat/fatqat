---
title: "Qiskit"
---

# Qiskit

<a id="fatqat.qiskit"></a>

A Qiskit circuit can run in FatQat through either API. Use
[`circuit_to_program()`][fatqat.qiskit.circuit_to_program] when you want a FatQat
program and want to read results in FatQat. Use
[`FatqatBackend`][fatqat.qiskit.FatqatBackend] to keep working with Qiskit's
transpiler, jobs, and results while FatQat runs the circuit. Both routes use
`fatqat.qiskit` and require Qiskit to be installed.

## Convert a circuit { #circuit-conversion }

Bind the parameters before converting the circuit. If the circuit contains gates
outside the basis returned by
[`build_simulator_target()`][fatqat.qiskit.build_simulator_target], transpile it
to that basis first. The converter keeps named registers and stores the circuit's
global phase in `program.metadata["qiskit_global_phase"]`. The returned state
omits that overall phase.

## Run through Qiskit { #backend-execution }

Transpile the circuit for the backend's `target`. To include noise, pass a FatQat
[`NoiseModel`][fatqat.NoiseModel] as `noise_model` when you create the backend.
Then submit one or more circuits with [`run()`][fatqat.qiskit.FatqatBackend.run].
Its reference lists the settings for shot count, measurement memory, and the
random seed.

If `run_input` is empty or contains non-circuit values, or if a run option is
invalid, the backend raises
[`QiskitBackendError`][fatqat.qiskit.QiskitBackendError] before returning a job.
If conversion or execution fails after that,
[`FatqatJob.result()`][fatqat.qiskit.FatqatJob.result] raises a Qiskit
`QiskitError` and retains the original exception as its cause.

<a id="jobs-and-provider-helper"></a>

[`FatqatProvider`][fatqat.qiskit.FatqatProvider] creates configured backends for
code that uses a provider interface.

## Compare bit and basis order

In a single two-bit classical register, a 1 in bit 0 appears as `"10"` in FatQat
and `"01"` in Qiskit. With multiple registers, read the groups in the Qiskit count
string separately. FatQat's
[`get_counts_as_tuples()`][fatqat.Result.get_counts_as_tuples] gives each outcome
as a tuple in classical register order.

For the same two qubits, Z on qubit 0 is `"ZI"` in FatQat and `"IZ"` in Qiskit. To
compare statevectors, reshape each into one axis per qubit, then reverse the axis
order in one of the arrays. For a matrix, do this separately for the qubit axes on
the row side and the column side.

## API reference { #reference }

::: fatqat.qiskit.circuit_to_program

::: fatqat.qiskit.FatqatBackend
    options:
      members: [target, max_circuits, coupling_map, run]
      inherited_members: false
      show_bases: false

::: fatqat.qiskit.FatqatJob
    options:
      members: [status, result, submit, cancel, backend]
      inherited_members: false
      show_bases: false

::: fatqat.qiskit.FatqatProvider
    options:
      members: [backends, get_backend]
      inherited_members: false
      show_bases: false

::: fatqat.qiskit.build_simulator_target

::: fatqat.qiskit.QiskitConversionError
    options:
      members: false
      inherited_members: false
      show_bases: false

::: fatqat.qiskit.QiskitBackendError
    options:
      members: false
      inherited_members: false
      show_bases: false
