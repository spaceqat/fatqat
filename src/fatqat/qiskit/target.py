"""Qiskit ``Target`` construction for fatqat simulators."""

from __future__ import annotations

from qiskit.circuit import Barrier, Measure, Reset
from qiskit.circuit.library import (
    CCXGate,
    CSGate,
    CSwapGate,
    CXGate,
    CYGate,
    CZGate,
    CPhaseGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    SXGate,
    SwapGate,
    TGate,
    TdgGate,
    U1Gate,
    U2Gate,
    U3Gate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.transpiler import Target

try:
    from qiskit.circuit.library import iSwapGate
except ImportError:  # pragma: no cover
    iSwapGate = None


def build_simulator_target() -> Target:
    """Build an unbounded ideal gate-level simulator target."""
    target = Target(description="fatqat gate-level simulator", num_qubits=None)
    instructions = [
        IGate(),
        HGate(),
        XGate(),
        YGate(),
        ZGate(),
        SGate(),
        SdgGate(),
        SXGate(),
        TGate(),
        TdgGate(),
        RXGate(0.0),
        RYGate(0.0),
        RZGate(0.0),
        PhaseGate(0.0),
        UGate(0.0, 0.0, 0.0),
        U1Gate(0.0),
        U2Gate(0.0, 0.0),
        U3Gate(0.0, 0.0, 0.0),
        CXGate(),
        CYGate(),
        CZGate(),
        CPhaseGate(0.0),
        CSGate(),
        SwapGate(),
        CCXGate(),
        CSwapGate(),
        Measure(),
        Reset(),
        Barrier(1),
    ]
    if iSwapGate is not None:
        instructions.append(iSwapGate())
    for instruction in instructions:
        target.add_instruction(instruction)
    return target
