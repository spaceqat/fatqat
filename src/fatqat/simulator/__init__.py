"""Gate-level simulators.

``Simulator`` is the general matrix backend for statevector, density-matrix,
unitary, and super-operator methods. The superconducting subclasses add fixed
grid gate sets. ``AtomArraySimulator`` adds dynamic pairing, occupancy, and
loss. Pulse-resolved models are available from ``fatqat.emulator``.
"""

from __future__ import annotations

from .fake_atom_array import AtomArraySimulator
from .fake_superconducting import SCQubitSimulator
from .simulator import Simulator
from .custom_operations import ExecutionContext
from ._engine.state import ClassicalState, EvolutionState

__all__ = [
    "Simulator",
    "ExecutionContext",
    "ClassicalState",
    "EvolutionState",
    "AtomArraySimulator",
    "SCQubitSimulator",
]
