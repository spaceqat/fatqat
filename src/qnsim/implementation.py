"""Class-keyed matrix implementations and the flat payload the engine consumes.

A matrix implementation maps an operation to its local matrix (physics only).
The backend pairs that matrix with layout-resolved target indices to build an
``ApplyMatrixStep`` — the plain data container the statevector engine reads
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import operations as ops
from .operations import Operation
from .program import AppliedOperation

MatrixImplementation = Callable[[AppliedOperation], np.ndarray]


@dataclass(frozen=True)
class ApplyMatrixStep:
    """Resolved local matrix payload consumed by the statevector engine.

    Doubles as the "apply a matrix" entry in a backend execution plan and as the
    payload the engine applies. The matrix is marked read-only after construction
    so this frozen value object cannot be mutated through the NumPy array buffer.

    Attributes:
        matrix: Local operation matrix.
        target_indices: Flat qubit indices the matrix acts on.
    """

    matrix: np.ndarray
    target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        # The engine consumes the matrix read-only; lock it so this frozen
        # dataclass is truly immutable (Python cannot freeze array contents).
        self.matrix.flags.writeable = False


# Module-level constant matrices (reused; do not rebuild per call).
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
# 2-qubit, operand 0 = MSB (control), operand 1 = LSB (target).
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _rx(applied: AppliedOperation) -> np.ndarray:
    """Build the RX matrix from the applied operation's angle."""
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(applied: AppliedOperation) -> np.ndarray:
    """Build the RY matrix from the applied operation's angle."""
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(applied: AppliedOperation) -> np.ndarray:
    """Build the RZ matrix from the applied operation's angle."""
    theta = applied.operation.theta
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


class MatrixImplementationMap:
    """Class-keyed registry from operation classes to matrix implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map."""
        self._rules: dict[type[Operation], MatrixImplementation] = {}

    def register(self, op_cls: type[Operation], rule: MatrixImplementation) -> None:
        """Register a matrix implementation for an operation class.

        Args:
            op_cls: Operation class used as the lookup key.
            rule: Callable that receives an `AppliedOperation` and returns a
                local matrix.
        """
        self._rules[op_cls] = rule

    def get(self, op_cls: type[Operation]) -> MatrixImplementation | None:
        """Return the matrix implementation for an operation class, if registered."""
        return self._rules.get(op_cls)


def default_implementation_map() -> MatrixImplementationMap:
    """Build the Phase 1 matrix implementation map."""
    m = MatrixImplementationMap()
    m.register(ops.XGate, lambda _ao: _X)
    m.register(ops.YGate, lambda _ao: _Y)
    m.register(ops.ZGate, lambda _ao: _Z)
    m.register(ops.HGate, lambda _ao: _H)
    m.register(ops.TGate, lambda _ao: _T)
    m.register(ops.CXGate, lambda _ao: _CX)
    m.register(ops.CZGate, lambda _ao: _CZ)
    m.register(ops.RX, _rx)
    m.register(ops.RY, _ry)
    m.register(ops.RZ, _rz)
    return m
