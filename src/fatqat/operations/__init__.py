"""Built-in gates, measurements, and structural operations.

Applications should use ``import fatqat.operations as ops``.
"""

from __future__ import annotations

# Prefer Barrier, Reset, Put, Pair, and Unpair over their backing `*Gate`
# classes, which are internal implementation details.
from .base import Operation
from .barrier import Barrier
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
from .measurement import Measurement
from .pairing import Pair, Unpair
from .parametric_gates import CPhase, Phase, RX, RY, RZ, U, U1, U2, U3
from .put import Put
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
from .reset import Reset

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
    "U",
    "U1",
    "U2",
    "U3",
    "CPhase",
    "Reset",
    "Barrier",
    "Put",
    "Pair",
    "Unpair",
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
