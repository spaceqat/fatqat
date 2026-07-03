"""Tests matrix implementation rules and immutable matrix payloads."""

import numpy as np
import pytest

from qnsim import operations as ops
from qnsim.implementation import (
    ApplyMatrixStep,
    MatrixImplementationMap,
    default_implementation_map,
)
from qnsim.program import AppliedOperation
from qnsim.registers import QuantumRegister


def _applied(op, n=2):
    qr = QuantumRegister(n)
    targets = tuple(qr[i] for i in range(op.num_qubits))
    return AppliedOperation(operation=op, targets=targets)


def test_fixed_gate_matrices():
    m = default_implementation_map()
    x = m.get(type(ops.X))(_applied(ops.X))
    assert np.allclose(x, [[0, 1], [1, 0]])
    cz = m.get(type(ops.CZ))(_applied(ops.CZ))
    assert np.allclose(cz, np.diag([1, 1, 1, -1]))
    cx = m.get(type(ops.CX))(_applied(ops.CX))
    assert np.allclose(cx, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def test_h_matrix_is_unitary_and_correct():
    m = default_implementation_map()
    h = m.get(type(ops.H))(_applied(ops.H))
    assert np.allclose(h, np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_batch1_fixed_single_qubit_gate_matrices():
    m = default_implementation_map()
    i_matrix = m.get(type(ops.I))(_applied(ops.I))
    assert np.allclose(i_matrix, np.eye(2))
    s = m.get(type(ops.S))(_applied(ops.S))
    assert np.allclose(s, [[1, 0], [0, 1j]])
    sdg = m.get(type(ops.Sdg))(_applied(ops.Sdg))
    assert np.allclose(sdg, [[1, 0], [0, -1j]])
    tdg = m.get(type(ops.Tdg))(_applied(ops.Tdg))
    assert np.allclose(tdg, [[1, 0], [0, np.exp(-1j * np.pi / 4)]])


def test_parametric_rx_reads_theta():
    m = default_implementation_map()
    theta = 0.5
    rx = m.get(ops.RX)(_applied(ops.RX(theta)))
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    assert np.allclose(rx, [[c, -1j * s], [-1j * s, c]])


def test_parametric_phase_reads_theta():
    m = default_implementation_map()
    theta = 0.9
    phase = m.get(ops.Phase)(_applied(ops.Phase(theta)))
    assert np.allclose(phase, [[1, 0], [0, np.exp(1j * theta)]])


def test_unregistered_class_returns_none():
    m = MatrixImplementationMap()
    assert m.get(type(ops.X)) is None


def test_apply_matrix_step_value_object():
    step = ApplyMatrixStep(matrix=np.eye(2, dtype=complex), target_indices=(3,))
    assert step.target_indices == (3,)
    with pytest.raises(ValueError):
        step.matrix[0, 0] = 5.0
