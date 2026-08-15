"""Operation base class and built-in gates.

Applications should use ``import fatqat.operations as op``.
"""

from __future__ import annotations

from .base import Operation
from .barrier import Barrier, BarrierGate
from .fixed_gates import (
    CCX,
    CS,
    CSwap,
    CX,
    CY,
    CZ,
    H,
    I,
    S,
    Sdg,
    SX,
    Swap,
    T,
    Tdg,
    X,
    Y,
    Z,
    iSwap,
)
from .load_atom import LoadAtoms
from .measurement import Measurement
from .parametric_gates import CPhase, Phase, RX, RY, RZ
from .pulse import PulseOperation
from .qudit_gates import (
    CClock,
    Clock,
    Fourier,
    InverseFourier,
    Shift,
    Sum,
    SubspaceRX,
    SubspaceRY,
    SubspaceRZ,
    SwapLevels,
)
from .rearrange import Rearrange
from .reset import Reset, ResetGate
from .refill import Refill, RefillGate

__all__ = [
    "Operation",
    "I",
    "H",
    "S",
    "Sdg",
    "SX",
    "T",
    "Tdg",
    "X",
    "Y",
    "Z",
    "CX",
    "CZ",
    "Swap",
    "CY",
    "CS",
    "iSwap",
    "CCX",
    "CSwap",
    "RX",
    "RY",
    "RZ",
    "Phase",
    "CPhase",
    "Reset",
    "Refill",
    "Barrier",
    "LoadAtoms",
    "Rearrange",
    "Measurement",
    "Shift",
    "Clock",
    "Sum",
    "SwapLevels",
    "Fourier",
    "InverseFourier",
    "SubspaceRX",
    "SubspaceRY",
    "SubspaceRZ",
    "CClock",
    "PulseOperation",
]
