"""Two-level neutral-atom pulse emulator and immutable model values.

The concrete backend binds structural global controls to an arrangement and
supports the same optional gate and Lindblad implementation-map paths as the
other pulse-emulator families.
"""

from .backend import Atom2LevelEmulator
from .model import Atom2LevelModel

__all__ = [
    "Atom2LevelEmulator",
    "Atom2LevelModel",
]
