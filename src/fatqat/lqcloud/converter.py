"""Translate the FATQAT QEC17 native circuit language to LQCloud."""

from __future__ import annotations

from importlib import import_module
from math import isfinite
from numbers import Real
from typing import Any

from .. import operations as ops
from .._backends.view_normalization import _break_grouped_operations
from ..program import Program, _AppliedOperation
from ..registers import RegisterRef
from ._qec17 import NUM_QUBITS

_FIXED_NATIVE_METHODS = {
    ops.I: "id",
    ops.H: "h",
    ops.HY: "hy",
    ops.X: "x",
    ops.MX: "mx",
    ops.Y: "y",
    ops.MY: "my",
    ops.Z: "z",
    ops.MZ: "mz",
    ops.XHalf: "xhalf",
    ops.MXHalf: "mxhalf",
    ops.YHalf: "yhalf",
    ops.MYHalf: "myhalf",
    ops.XYHalf: "xyhalf",
    ops.MXYHalf: "mxyhalf",
    ops.MXMYHalf: "mxmyhalf",
    ops.XMYHalf: "xmyhalf",
    ops.S: "s",
    ops.Sdg: "sdg",
    ops.T: "t",
}


def _load_lqcloud() -> Any:
    """Load the optional SDK without making this namespace import-dependent."""
    try:
        return import_module("lqcloud")
    except ModuleNotFoundError as exc:
        if exc.name != "lqcloud":
            raise
        raise ImportError(
            "the LQCloud integration requires the optional SDK; "
            "install it with 'pip install lqcloud==0.4.2'"
        ) from exc


def _flatten_quantum_refs(program: Program) -> tuple[dict[RegisterRef, int], int]:
    """Allocate program qubits in register declaration order."""
    if not isinstance(program, Program):
        raise TypeError(f"program must be a fatqat.Program, got {type(program)!r}")

    refs: dict[RegisterRef, int] = {}
    for register in program.quantum_registers:
        if register.dim != 2:
            raise ValueError(
                "LQCloud QZ01-surface_code accepts qubits with dim=2; "
                f"register {register.name!r} has dim={register.dim}"
            )
        for index in range(register.size):
            refs[register[index]] = len(refs)

    n_qubits = len(refs)
    if n_qubits == 0:
        raise ValueError("LQCloud QZ01-surface_code requires at least one qubit")
    if n_qubits > NUM_QUBITS:
        raise ValueError(
            "LQCloud QZ01-surface_code supports at most "
            f"{NUM_QUBITS} qubits, got {n_qubits}"
        )
    return refs, n_qubits


def _unsupported_operation(operation: ops.Operation) -> ValueError:
    if operation is ops.Reset:
        return ValueError(
            "LQCloud QEC17 static circuits do not accept explicit Reset; "
            "initial state preparation is owned by the LQCloud platform"
        )
    return ValueError(
        f"operation {operation.name!r} is not an LQCloud QEC17 native operation; "
        "compile or rewrite the program to the QEC17 native gate set first"
    )


def program_to_qec17_circuit(program: Program) -> Any:
    """Convert one static FATQAT program to an LQCloud QEC17 circuit.

    Quantum registers are flattened in declaration order. Only the QEC17
    native gate values are translated; this function performs no compilation.
    FATQAT measurements, explicit resets, conditions, and unsupported gates
    are rejected. One full-width barrier and ``measure_all`` are appended to
    the returned circuit.

    Args:
        program: A bound Program with 1 to 17 dimension-two qubits, QEC17
            native gates, and optional barriers. Classical conditions,
            explicit resets, and measurements are rejected.

    Returns:
        Any: An LQCloud QuantumCircuit with a terminal full barrier and
            measurement of every logical qubit. No connection or submission
            is made. Physical CZ adjacency is checked by
            ``LQCloudQEC17Backend.run``, not here.

    Raises:
        ImportError: If the optional LQCloud SDK is not installed.
        TypeError: If ``program`` is not a `fatqat.Program`.
        ValueError: If the program exceeds QEC17's supported shape or contains
            measurement, reset, feedforward, an unbound RZ angle, or any gate
            outside the exact native set.
    """
    refs, n_qubits = _flatten_quantum_refs(program)
    lqcloud_sdk = _load_lqcloud()
    circuit = lqcloud_sdk.QuantumCircuit(n_qubits)

    for step in _break_grouped_operations(program._instructions):
        if isinstance(step, ops.Measurement):
            raise ValueError(
                "LQCloud QEC17 static circuits do not accept FATQAT Measurement "
                "instructions; a final barrier and measure_all are added automatically"
            )
        if not isinstance(step, _AppliedOperation):  # pragma: no cover - invariant
            raise TypeError(f"unsupported FATQAT instruction {step!r}")
        if step.condition is not None:
            raise ValueError(
                f"conditional operation {step.operation.name!r} is not supported; "
                "the LQCloud QEC17 integration does not support dynamic circuits"
            )

        operation = step.operation
        targets = tuple(refs[target] for target in step.targets)
        method_name = next(
            (
                name
                for native_operation, name in _FIXED_NATIVE_METHODS.items()
                if operation is native_operation
            ),
            None,
        )
        if method_name is not None:
            getattr(circuit, method_name)(targets[0])
        elif type(operation) is ops.RZ:
            theta = operation.theta
            if isinstance(theta, bool) or not isinstance(theta, Real):
                raise ValueError(
                    "RZ has an unbound or non-real parameter; bind the FATQAT "
                    "program before converting it to LQCloud"
                )
            try:
                numeric_theta = float(theta)
            except OverflowError:
                raise ValueError(
                    "RZ theta must be finite for LQCloud; its magnitude is too large"
                ) from None
            if not isfinite(numeric_theta):
                raise ValueError(
                    f"RZ theta must be finite for LQCloud, got {numeric_theta!r}"
                )
            circuit.rz(numeric_theta, targets[0])
        elif type(operation) is ops.SU2:
            circuit.su2(operation.matrix, targets[0])
        elif operation is ops.CZ:
            circuit.cz(targets[0], targets[1])
        elif operation is ops.Barrier:
            circuit.barrier(*targets)
        else:
            raise _unsupported_operation(operation)

    circuit.barrier()
    circuit.measure_all()
    return circuit


__all__ = ["program_to_qec17_circuit"]
