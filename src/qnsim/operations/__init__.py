"""Operation base class and the built-in gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from .base import Operation
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
from .qudit_gates import Clock, Shift, Sum, SumGate, SwapLevels
from .reset import Reset, ResetGate

__all__ = [
    "Operation",
    "I", "H", "S", "Sdg", "T", "Tdg", "X", "Y", "Z",
    "CX", "CZ", "Swap", "CY", "CS", "iSwap", "CCX", "CSwap",
    "RX", "RY", "RZ", "Phase",
    "CPhase",
    "Reset",
    "Measurement",
    "Shift", "Clock", "Sum", "SwapLevels",
]
