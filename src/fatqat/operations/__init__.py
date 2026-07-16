"""Operation base class and the built-in gate set, exposed as the `fq.ops` namespace."""

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
from .parametric_gates import CPhase, Phase, RX, RY, RZ
from .qudit_gates import (
    CClock,
    Clock,
    Fourier,
    Fourierdg,
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
    "CPhase",
    "Reset",
    "Barrier",
    "Measurement",
    "Shift",
    "Clock",
    "Sum",
    "SwapLevels",
    "Fourier",
    "Fourierdg",
    "SubspaceRX",
    "SubspaceRY",
    "SubspaceRZ",
    "CClock",
]
