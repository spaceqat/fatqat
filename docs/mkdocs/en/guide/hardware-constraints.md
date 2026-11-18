# Simulate with hardware constraints

A circuit can produce the expected result and still be unsuitable for a device.
The device may support only some gates, or require two qubits to occupy connected
positions.

Superconducting and neutral-atom gate simulations still use matrix-based
methods to calculate how the quantum state changes. For example, a statevector
or density matrix represents the state, and gates and noise channels act on it.
The simulator also checks device rules: is the operation supported, are the
target positions connected, and have the required atoms been loaded?

For example, a two-qubit gate may run in the general simulator but fail on a
superconducting device model because the selected positions are not connected.
This makes the constraint visible during simulation, so you can adjust the
Program or qubit placement.

This section covers superconducting devices, neutral atoms, and linear optics.
Linear optics represents photons in optical modes and can use a specialized
fixed-photon-number representation to reduce the computation. Use Hamiltonian
emulation when you need to study how pulse shape and duration affect the
quantum state.

## Examples and compilation

- [Superconducting qubits](superconducting-simulation.md)
- [Neutral-atom arrays](atom-array-simulation.md)
- [Linear optics](linear-optics.md)
- [Compile circuits](compiler.md)
