"""Construct the Qiskit target advertised by the FATQAT adapter."""

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
    UGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.transpiler import Target

# The legacy u1/u2/u3 gates are deprecated in Qiskit and are the realistic
# future removal risk, so only they are import-guarded; iSwapGate has been in
# qiskit.circuit.library since before 1.0.
try:
    from qiskit.circuit.library import U1Gate, U2Gate, U3Gate
except ImportError:  # pragma: no cover
    U1Gate = U2Gate = U3Gate = None


def build_simulator_target() -> Target:
    """Return a new unbounded, fully connected gate-level target.

    The target contains the fixed and parameterized qubit gates understood by
    :func:`~fatqat.qiskit.circuit_to_program`, together with measurement,
    reset, and barrier. The legacy ``u1``/``u2``/``u3`` gates are included
    when the installed Qiskit version still provides their (deprecated)
    classes.
    """
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
        iSwapGate(),
    ]
    if U1Gate is not None:
        instructions += [U1Gate(0.0), U2Gate(0.0, 0.0), U3Gate(0.0, 0.0, 0.0)]
    for instruction in instructions:
        target.add_instruction(instruction)
    return target
