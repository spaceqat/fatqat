"""Gate-level simulators.

:class:`Simulator` is the general matrix backend for statevector,
density-matrix, unitary, and super-operator methods. The superconducting
subclasses add fixed grid gate sets; :class:`AtomArraySimulator` adds a native
gate set, dynamic pairing, occupancy, and loss. Pulse-resolved models live in
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
