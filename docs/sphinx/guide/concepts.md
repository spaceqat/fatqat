# Concepts

qnsim's model has five moving parts, and each one owns exactly one job.

## Register and RegisterRef

A {py:class}`~qnsim.Register` is a fixed-size block of quantum or classical
storage — a {py:class}`~qnsim.QuantumRegister` or
{py:class}`~qnsim.ClassicalRegister`. Every slot has a dimension (`dim`,
default 2 for a qubit); higher dimensions describe a qudit.

Indexing a register gives a {py:class}`~qnsim.RegisterRef`, a reference to
one slot. `Program` accepts bare integers as shorthand for a `RegisterRef`
when there's only one register of that kind to be unambiguous about —
otherwise you pass an explicit ref such as `program.qreg[0][1]`.

## Program

A {py:class}`~qnsim.Program` owns a program's registers and its ordered list
of instructions. It doesn't execute anything itself — `add()` and
`add_measurement()` just validate and append. Operations run in the order
they were inserted.

## Operation and Measurement

An `Operation` (see [Gates](gates.md)) describes *what* to do — a gate, not
tied to any qubit. Calling `program.add(op, targets)` resolves the targets
against the program's registers and records the operation bound to those
targets. `Measurement` is a separate instruction type recorded by
`add_measurement()` — a readout from quantum refs into matching classical
refs. Both live side by side in `program.operations`, in insertion order.

## Backend, Job, Result

A backend (currently {py:class}`~qnsim.backends.StateVectorBackend`) is what
actually executes a `Program`. Its `run()` method returns a
{py:class}`~qnsim.Job` — an eager, already-completed handle — whose
`result()` yields a {py:class}`~qnsim.Result` or re-raises an execution
error. `Result` exposes whichever fields the run actually produced (counts,
a statevector, or both); see [Running and results](running-and-results.md)
for how that's controlled and read back.
