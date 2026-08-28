"""Convert static Qiskit circuits into FATQAT programs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import fatqat.operations as ops
from fatqat.program import Program
from fatqat.registers import ClassicalRegister, QuantumRegister, RegisterRef

from .errors import QiskitConversionError

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def circuit_to_program(circuit: QuantumCircuit) -> Program:
    """Convert one bound, static Qiskit circuit into a FATQAT program.

    Supported instructions are measurement, reset, barrier, and the gate
    basis declared by :func:`~fatqat.qiskit.build_simulator_target`. Barrier
    instructions are discarded. Transpile other gate-level circuits to that
    target before conversion.

    The converter preserves named quantum and classical registers. It records
    the circuit name, metadata, and global phase in the program metadata, but
    the global phase is not applied during FATQAT execution. Standalone bits,
    unbound instruction parameters, delay or pulse instructions, and classical
    control flow are not supported.

    Args:
        circuit: Bound Qiskit ``QuantumCircuit`` to convert.

    Returns:
        A dimension-2 :class:`~fatqat.Program` with equivalent supported
        instructions.

    Raises:
        QiskitConversionError: If the circuit uses an unsupported instruction,
            control flow, standalone bit, unbound instruction parameter, or
            unbound global phase.
        TypeError: If ``circuit`` is not a ``QuantumCircuit``.
    """
    from qiskit.circuit import (
        ControlFlowOp,
        Measure,
        QuantumCircuit,
        Reset,
    )

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"expected qiskit.QuantumCircuit, got {type(circuit).__name__!r}"
        )

    _reject_control_flow(circuit)
    program = _program_from_registers(circuit)
    metadata = dict(program.metadata)
    metadata["qiskit_global_phase"] = _bound_global_phase(circuit)
    if circuit.metadata:
        metadata["qiskit_circuit_metadata"] = dict(circuit.metadata)
    program.metadata.update(metadata)

    for instruction in circuit.data:
        inst_op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        name = inst_op.name

        if isinstance(inst_op, ControlFlowOp):
            raise QiskitConversionError(
                f"circuit {circuit.name!r}: classical control flow "
                f"({name!r}) is not supported by the fatqat Qiskit adapter"
            )

        if name == "barrier":
            continue

        if name == "delay":
            raise QiskitConversionError(
                f"circuit {circuit.name!r}: delay/pulse instructions "
                f"({name!r}) are not supported"
            )

        if isinstance(inst_op, Measure):
            if not qubits or not clbits or len(qubits) != len(clbits):
                raise QiskitConversionError(
                    f"circuit {circuit.name!r}: invalid measure arity "
                    f"({len(qubits)} qubits, {len(clbits)} clbits)"
                )
            q_targets = tuple(
                _classical_or_quantum_ref(program, circuit, q, quantum=True)
                for q in qubits
            )
            c_outputs = tuple(
                _classical_or_quantum_ref(program, circuit, c, quantum=False)
                for c in clbits
            )
            program.measure(q_targets, c_outputs)
            continue

        if isinstance(inst_op, Reset):
            if len(qubits) != 1:
                raise QiskitConversionError(
                    f"circuit {circuit.name!r}: reset expects one qubit, "
                    f"got {len(qubits)}"
                )
            program.add(
                ops.Reset,
                _classical_or_quantum_ref(program, circuit, qubits[0], quantum=True),
            )
            continue

        params = _bound_params(inst_op, circuit.name)
        if name in _FIXED_1Q:
            program.add(
                _FIXED_1Q[name],
                _single_qubit_ref(program, circuit, qubits),
            )
            continue
        if name in _FIXED_2Q:
            program.add(
                _FIXED_2Q[name],
                _two_qubit_targets(program, circuit, qubits),
            )
            continue
        if name in _FIXED_3Q:
            program.add(
                _FIXED_3Q[name],
                _three_qubit_targets(program, circuit, qubits),
            )
            continue
        if name in _PARAMETRIC:
            expected, factory = _PARAMETRIC[name]
            if len(params) != expected:
                raise QiskitConversionError(
                    f"circuit {circuit.name!r}: {name!r} expects {expected} "
                    f"parameter(s), got {len(params)}"
                )
            if expected == 1 and name in {"rx", "ry", "rz", "p", "phase", "u1"}:
                program.add(
                    factory(params),
                    _single_qubit_ref(program, circuit, qubits),
                )
                continue
            if expected == 2 and name == "u2":
                program.add(
                    factory(params),
                    _single_qubit_ref(program, circuit, qubits),
                )
                continue
            if expected == 3 and name in {"u", "u3"}:
                program.add(
                    factory(params),
                    _single_qubit_ref(program, circuit, qubits),
                )
                continue
            if expected == 1 and name in {"cp", "cu1"}:
                program.add(
                    factory(params),
                    _two_qubit_targets(program, circuit, qubits),
                )
                continue

        raise QiskitConversionError(
            f"circuit {circuit.name!r}: unsupported instruction {name!r} "
            f"({type(inst_op).__name__}); transpile to the backend target basis first"
        )

    return program


_FIXED_1Q = {
    "id": ops.I,
    "h": ops.H,
    "x": ops.X,
    "y": ops.Y,
    "z": ops.Z,
    "s": ops.S,
    "sdg": ops.Sdg,
    "sx": ops.SX,
    "t": ops.T,
    "tdg": ops.Tdg,
}

_FIXED_2Q = {
    "cx": ops.CX,
    "cy": ops.CY,
    "cz": ops.CZ,
    "cs": ops.CS,
    "swap": ops.Swap,
    "iswap": ops.iSwap,
}

_FIXED_3Q = {
    "ccx": ops.CCX,
    "cswap": ops.CSwap,
}

_PARAMETRIC: dict[str, tuple[int, Callable[[list[float]], Any]]] = {
    "rx": (1, lambda p: ops.RX(p[0])),
    "ry": (1, lambda p: ops.RY(p[0])),
    "rz": (1, lambda p: ops.RZ(p[0])),
    "p": (1, lambda p: ops.Phase(p[0])),
    "phase": (1, lambda p: ops.Phase(p[0])),
    "u1": (1, lambda p: ops.U1(p[0])),
    "u2": (2, lambda p: ops.U2(p[0], p[1])),
    "u": (3, lambda p: ops.U(p[0], p[1], p[2])),
    "u3": (3, lambda p: ops.U3(p[0], p[1], p[2])),
    "cp": (1, lambda p: ops.CPhase(p[0])),
    "cu1": (1, lambda p: ops.CPhase(p[0])),
}


def _reject_control_flow(circuit: QuantumCircuit) -> None:
    from qiskit.circuit import ControlFlowOp

    for instruction in circuit.data:
        if isinstance(instruction.operation, ControlFlowOp):
            raise QiskitConversionError(
                f"circuit {circuit.name!r}: classical control flow is not supported; "
                "measurement and reset are supported, but branching, looping, and "
                "feedforward are not"
            )


def _program_from_registers(circuit: QuantumCircuit) -> Program:
    qregs = [QuantumRegister(qreg.size, name=qreg.name) for qreg in circuit.qregs]
    cregs = [ClassicalRegister(creg.size, name=creg.name) for creg in circuit.cregs]
    return Program(
        qregs if qregs else 0,
        cregs if cregs else 0,
        metadata={"qiskit_circuit_name": circuit.name},
    )


def _fatqat_register(
    program: Program,
    qiskit_register: Any,
    *,
    quantum: bool,
) -> QuantumRegister | ClassicalRegister:
    registers = program.quantum_registers if quantum else program.classical_registers
    for reg in registers:
        if reg.name == qiskit_register.name and reg.size == qiskit_register.size:
            return reg
    kind = "quantum" if quantum else "classical"
    raise QiskitConversionError(
        f"circuit register {qiskit_register.name!r} does not match program {kind} "
        "registers"
    )


def _classical_or_quantum_ref(
    program: Program,
    circuit: QuantumCircuit,
    bit: Any,
    *,
    quantum: bool,
) -> RegisterRef:
    loc = circuit.find_bit(bit)
    if not loc.registers:
        kind = "qubit" if quantum else "classical bit"
        raise QiskitConversionError(
            f"circuit {circuit.name!r}: standalone {kind} not owned by a register "
            "is not supported by the fatqat Qiskit adapter"
        )
    qiskit_reg, index = loc.registers[0]
    reg = _fatqat_register(program, qiskit_reg, quantum=quantum)
    return reg[index]


def _single_qubit_ref(
    program: Program, circuit: QuantumCircuit, qubits: tuple[Any, ...]
) -> RegisterRef:
    if len(qubits) != 1:
        raise QiskitConversionError(f"expected one qubit operand, got {len(qubits)}")
    return _classical_or_quantum_ref(program, circuit, qubits[0], quantum=True)


def _two_qubit_targets(
    program: Program, circuit: QuantumCircuit, qubits: tuple[Any, ...]
) -> tuple[RegisterRef, RegisterRef]:
    if len(qubits) != 2:
        raise QiskitConversionError(f"expected two qubit operands, got {len(qubits)}")
    return (
        _classical_or_quantum_ref(program, circuit, qubits[0], quantum=True),
        _classical_or_quantum_ref(program, circuit, qubits[1], quantum=True),
    )


def _three_qubit_targets(
    program: Program, circuit: QuantumCircuit, qubits: tuple[Any, ...]
) -> tuple[RegisterRef, RegisterRef, RegisterRef]:
    if len(qubits) != 3:
        raise QiskitConversionError(f"expected three qubit operands, got {len(qubits)}")
    return (
        _classical_or_quantum_ref(program, circuit, qubits[0], quantum=True),
        _classical_or_quantum_ref(program, circuit, qubits[1], quantum=True),
        _classical_or_quantum_ref(program, circuit, qubits[2], quantum=True),
    )


def _bound_params(inst_op: Any, circuit_name: str) -> list[float]:
    from qiskit.circuit import ParameterExpression

    values: list[float] = []
    for param in inst_op.params:
        if isinstance(param, ParameterExpression):
            if not param.parameters:
                values.append(float(param))
                continue
            names = ", ".join(sorted(str(name) for name in param.parameters))
            raise QiskitConversionError(
                f"circuit {circuit_name!r}: instruction {inst_op.name!r} has "
                f"unbound parameter(s): {names}"
            )
        values.append(float(param))
    return values


def _bound_global_phase(circuit: QuantumCircuit) -> float:
    from qiskit.circuit import ParameterExpression

    phase = circuit.global_phase
    if not isinstance(phase, ParameterExpression) or not phase.parameters:
        return float(phase)

    try:
        return float(phase)
    except (TypeError, ValueError) as exc:
        names = ", ".join(sorted(str(name) for name in phase.parameters))
        raise QiskitConversionError(
            f"circuit {circuit.name!r}: global phase has unbound parameter(s): "
            f"{names}"
        ) from exc
