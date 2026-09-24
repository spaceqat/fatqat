"""Gate-level simulators.

``Simulator`` is the general matrix backend for statevector, density-matrix,
unitary, and super-operator methods. The superconducting subclasses add native
gate sets and device coupling graphs. ``AtomArraySimulator`` adds dynamic
pairing, occupancy, and loss. Pulse-resolved models are available from
``fatqat.emulator``.
"""

from __future__ import annotations

from .fake_atom_array import AtomArraySimulator
from .fake_superconducting import SCQubitQEC17Simulator, SCQubitSimulator
from .simulator import Simulator

__all__ = [
    "Simulator",
    "AtomArraySimulator",
    "SCQubitSimulator",
    "SCQubitQEC17Simulator",
]
