"""Private numerical execution layer for the gate-level `Simulator`.

`MatrixEngine` and its NumPy/Numba subclasses own the quantum state and the
numerics; the `Simulator` above them owns validation, lowering, and result
assembly. A `Simulator` constructs one engine and drives it.

This layer is private and deliberately re-exports nothing. A run crosses the
boundary as one immutable simulator-owned execution context and one resolved
policy. The engine configures dimensions, materializes an engine-specific
payload, and executes that payload only through local or shot-batch entry
points. The `Simulator` owns route dispatch and public result assembly, so
numeric engines never select process routes. Engines remain private rather
than a supported extension point; users reach simulation through
:class:`fatqat.simulator.Simulator`.

Import the concrete modules directly (``from ._engine.np import
NumpySVEngine``). ``nb`` is imported lazily by the `Simulator`, since numba
compilation is only needed when that runtime is selected.
"""
