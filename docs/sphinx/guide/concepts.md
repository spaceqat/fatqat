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

### GridRegister and RegisterView

A {py:class}`~fatqat.GridRegister` is a {py:class}`~fatqat.QuantumRegister`
subclass whose slots are arranged as a rectangular, row-major `rows x cols`
grid. It carries only that addressing structure — no physical-site,
placement, or calibration information. A `GridRegister` is a backend-neutral
frontend concept: it's constructed the same way regardless of which backend
eventually runs the program.

Its selection helpers — {py:meth}`~fatqat.GridRegister.all`,
{py:meth}`~fatqat.GridRegister.row`, {py:meth}`~fatqat.GridRegister.column`,
and {py:meth}`~fatqat.GridRegister.block` — don't return a tuple of refs.
They return a {py:class}`~fatqat.RegisterView`: an immutable, structured
target expression that names *which* members are selected without eagerly
expanding them. A view is only accepted where an operation opts in (see
[Gates](gates.md)); most operations still take a single
{py:class}`~fatqat.RegisterRef` or a fixed-arity tuple of them, exactly as
before.

Resolving a view against real hardware sites is a backend concern, not a
frontend one. `FakeAtomGridBackend`'s internal binder for this — a
`GridBinding` — figures out, once per run, how the program's `GridRegister`
maps onto that backend's device sites. `GridBinding` is not something a user
constructs, configures, or passes in; it never appears in the public API
surface, only inside the backend that resolves the program.

This split matters because of a numbering distinction that runs through the
whole feature: **noise selectors and implementation-map lookups use two
different numberings, and they are not always the same numbers.**
{py:meth}`~fatqat.NoiseModel.channels_for` and
{py:meth}`~fatqat.NoiseModel.readout_error_for` always operate in *flat
engine-index* space — the same indices used to build the simulator's state
vector, independent of any grid shape. An `ImplementationMap`
device-capability lookup (what
{py:attr}`~fatqat.backends.FakeAtomGridBackend.implementation_map` reports)
instead uses *backend device-site labels*: positions on the physical device
grid. For a plain, register-only program these two numberings coincide. But
once a smaller `GridRegister` runs on a larger backend — say a 2x3 grid
bound onto a 4x5 device — a program qubit's flat engine index and its
backend device label are generally different numbers referring to the same
qubit. Noise configuration should always be written in terms of flat engine
indices (or refs, which resolve to them); device labels are strictly an
internal detail of native-gate lookup and are never the right thing to use
when configuring noise.

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

A backend (currently {py:class}`~fatqat.backends.SimulatorBackend`) is what
actually executes a `Program`. Its
{py:meth}`~fatqat.backends.SimulatorBackend.run` method returns a
{py:class}`~fatqat.Job` — an eager, already-completed handle — whose
{py:meth}`~fatqat.Job.result` yields a {py:class}`~fatqat.Result` or
re-raises an execution error. `Result` exposes whichever fields the run
actually produced (counts, a statevector, or both); see
[Running and results](running-and-results.md) for how that's controlled and
read back.
