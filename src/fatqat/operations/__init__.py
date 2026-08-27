"""Built-in gates, measurements, and structural operations.

Applications should use ``import fatqat.operations as ops``. Names in
``__all__`` are the supported public surface. Gate-suffixed
implementation attributes, where present, are outside that surface.
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
from .measurement import Measurement
from .pairing import Pair, PairGate, Unpair, UnpairGate
from .parametric_gates import CPhase, Phase, RX, RY, RZ, U, U1, U2, U3
from .put import Put, PutGate
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
from .reset import Reset, ResetGate

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
