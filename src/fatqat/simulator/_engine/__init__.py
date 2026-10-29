"""Private numerical execution layer for the gate-level `Simulator`.

`MatrixEngine` and its NumPy/Numba subclasses own evolving quantum and classical
state and the numerics; the `Simulator` above them owns validation, lowering,
and result assembly. A `Simulator` constructs one engine and drives it.

This layer is private and deliberately re-exports nothing. A run crosses the
boundary as one immutable simulator-owned execution context and one resolved
policy. The engine configures dimensions, prepares an engine-specific
payload, and executes that payload only through local or worker-shot entry
points. Engines select execution policies and dispatch routes; the `Simulator`
assembles public results. Engines remain private rather than a supported
extension point; users reach simulation through
:class:`fatqat.simulator.Simulator`.

Import the concrete modules directly (``from ._engine.engines.numpy_engine import
NumpySVEngine``). The `Simulator` loads the selected runtime module lazily
when it is constructed.
"""
