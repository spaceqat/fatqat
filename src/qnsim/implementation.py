"""Class-keyed matrix rules. Rules return only the local matrix; indices come from layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import operations as ops
from .operations import Operation
from .program import AppliedOperation

MatrixRule = Callable[[AppliedOperation], np.ndarray]


@dataclass(frozen=True)
class MatrixImplementation:
    matrix: np.ndarray
    target_indices: tuple[int, ...]


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
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(applied: AppliedOperation) -> np.ndarray:
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(applied: AppliedOperation) -> np.ndarray:
    theta = applied.operation.theta
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


class MatrixImplementationMap:
    def __init__(self) -> None:
        self._rules: dict[type[Operation], MatrixRule] = {}

    def register(self, op_cls: type[Operation], rule: MatrixRule) -> None:
        self._rules[op_cls] = rule

    def get(self, op_cls: type[Operation]) -> MatrixRule | None:
        return self._rules.get(op_cls)


def default_implementation_map() -> MatrixImplementationMap:
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
