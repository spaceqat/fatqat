"""Gate-level simulators.

``Simulator`` is the general matrix backend for statevector, density-matrix,
unitary, and super-operator methods. The superconducting subclasses add fixed
grid gate sets. ``AtomArraySimulator`` adds dynamic pairing, occupancy, and
loss. ``LOSimulator`` truncates optical modes to a per-run photon cutoff.
Pulse-resolved models are available from ``fatqat.emulator``.
"""

from __future__ import annotations

from .fake_atom_array import AtomArraySimulator
from .fake_superconducting import SCQubitSimulator
from .linear_optics import LOSimulator, fixed_photon_index, fixed_photon_occupation
from .simulator import Simulator
from .custom_operations import ExecutionContext
from ._engine.state import BasisState, ClassicalState, EvolutionState

__all__ = [
    "Simulator",
    "LOSimulator",
    "fixed_photon_index",
    "fixed_photon_occupation",
    "ExecutionContext",
    "BasisState",
    "ClassicalState",
    "EvolutionState",
    "AtomArraySimulator",
    "SCQubitSimulator",
]
