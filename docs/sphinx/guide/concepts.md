# Concepts

fatqat's model has five moving parts, and each one owns exactly one job.

## Register and RegisterRef

A {py:class}`~fatqat.Register` is a fixed-size block of quantum or classical
storage — a {py:class}`~fatqat.QuantumRegister` or
{py:class}`~fatqat.ClassicalRegister`. Every slot has a dimension (`dim`,
default 2 for a qubit); higher dimensions describe a qudit.

Indexing a register gives a {py:class}`~fatqat.RegisterRef`, a reference to
one slot. {py:class}`~fatqat.Program` accepts bare integers as shorthand for
a {py:class}`~fatqat.RegisterRef` when there's only one register of that
kind to be unambiguous about — otherwise you pass an explicit ref such as
`program.qreg[0][1]`.

## Program

A {py:class}`~fatqat.Program` owns a program's registers and its ordered list
of instructions. It doesn't execute anything itself —
{py:meth}`~fatqat.Program.add` and {py:meth}`~fatqat.Program.add_measurement`
just validate and append. Operations run in the order they were inserted.

## Operation and Measurement

An {py:class}`~fatqat.operations.Operation` (see [Gates](gates.md)) describes *what*
to do — a gate, not tied to any qubit. Calling
{py:meth}`~fatqat.Program.add` resolves the targets against the program's
registers and records the operation bound to those targets.
{py:class}`~fatqat.Measurement` is a separate instruction type recorded by
{py:meth}`~fatqat.Program.add_measurement` — a readout from quantum refs into
matching classical refs. Both live side by side in
{py:attr}`~fatqat.Program.operations`, in insertion order.

## Backend, Job, Result

A backend (currently {py:class}`~fatqat.backends.StateVectorBackend`) is what
actually executes a `Program`. Its
{py:meth}`~fatqat.backends.StateVectorBackend.run` method returns a
{py:class}`~fatqat.Job` — an eager, already-completed handle — whose
{py:meth}`~fatqat.Job.result` yields a {py:class}`~fatqat.Result` or
re-raises an execution error. `Result` exposes whichever fields the run
actually produced (counts, a statevector, or both); see
[Running and results](running-and-results.md) for how that's controlled and
read back.
