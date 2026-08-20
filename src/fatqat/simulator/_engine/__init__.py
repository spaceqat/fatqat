"""Private numerical execution layer for the gate-level `Simulator`.

`MatrixEngine` and its NumPy/Numba subclasses own the quantum state and the
numerics; the `Simulator` above them owns validation, lowering, and result
assembly. A `Simulator` constructs one engine and drives it.

This layer is private and deliberately re-exports nothing. Its interface is
stated entirely in private types - ``MatrixEngine.run`` takes a
``list[ResolvedStep]``, an ``_EngineConfig``, and a result-request value
object, and returns a ``RawResult``, all of which live in
:mod:`fatqat._backends`. Engines are therefore not a supported extension
point: making them one would first require promoting that whole execution
contract to public API. Users reach simulation through
:class:`fatqat.simulator.Simulator`.

Import the concrete modules directly (``from ._engine.np import
NumpySVEngine``). ``nb`` is imported lazily by the `Simulator`, since numba
compilation is only needed when that runtime is selected.
"""
