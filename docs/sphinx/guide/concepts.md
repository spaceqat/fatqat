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
frontend one. The backend's per-run resource-map utility figures out how the
program's `GridRegister` maps onto that backend's device sites. This utility is not something a user
constructs, configures, or passes in; it never appears in the public API
surface, only inside the backend that resolves the program.

This split matters because of an identity distinction that runs through the
whole feature: **gate-channel noise selectors and implementation-map lookups
use two different identity spaces, and they are not always the same
values.** {py:meth}`~fatqat.NoiseModel.channels_for` matches a gate
occurrence's selector against either its *logical* {py:class}`~fatqat.RegisterRef`
targets (ref equality) or its *physical* device resource labels
(`resource_layout.device_operands(targets)` equality) — never against the
private flat engine index used internally to build the simulator's state
vector. An `ImplementationMap` device-capability lookup (what
{py:attr}`~fatqat.backends.FakeAtomGridBackend.implementation_map` reports)
also uses those same physical device-site labels: positions on the physical
device grid. For a plain, register-only program running on the generic
simulator, device labels and refs both coincide with declaration order. But
once a smaller `GridRegister` runs on a larger backend — say a 2x3 grid bound
onto a 4x5 device — a program qubit's device label is generally a different
number from its position in declaration order. A gate-channel noise selector
should be written either as the program's own `RegisterRef`s (logical) or as
the backend's device resource labels (physical); a bare integer is always
interpreted as a physical device label, never a flat engine index.
{py:meth}`~fatqat.NoiseModel.readout_error_for` has not yet been migrated to
this scheme and still operates in flat engine-index space — that
distinction is temporary, tracked as follow-up work.

### What's not supported yet

This is a first implementation, and it deliberately leaves several things
out. There's no custom or user-specified placement: binding a
{py:class}`~fatqat.GridRegister` onto a backend's device sites is entirely
internal, top-left corner first, row-major — a program cannot request a
different mapping. There's also no reshape support — no rotation,
transpose, rearrangement, or atom-transport operation for a `GridRegister`
once it's constructed, only the fixed layout it was built with. And a
program may bind at most one `GridRegister`; {py:class}`FakeAtomGridBackend
<fatqat.backends.FakeAtomGridBackend>` accepts exactly one `GridRegister` or
none, not several. These are areas for future work, not bugs.

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
