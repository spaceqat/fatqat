"""Gate-level simulation: the `Simulator` backend and its device specializations.

`Simulator` is the matrix-family entry point - ``Simulator(method="statevector")``
or ``Simulator(method="density_matrix")`` selects the state representation. The
`AtomArraySimulator` and `SCQubit*Simulator` classes specialize it with a fixed
device topology and native gate set.

The numerical execution layer lives in :mod:`fatqat.simulator._engine`
(`MatrixEngine` and its NumPy/Numba implementations); a `Simulator` owns one
engine instance and drives it. Pulse-level emulation is the sibling package
:mod:`fatqat.emulator`.
"""

from __future__ import annotations

from .fake_atom_array import AtomArraySimulator
from .fake_superconducting import SCQubitGoogleSimulator, SCQubitIBMSimulator
from .simulator import Simulator

__all__ = [
    "Simulator",
    "AtomArraySimulator",
    "SCQubitGoogleSimulator",
    "SCQubitIBMSimulator",
]
