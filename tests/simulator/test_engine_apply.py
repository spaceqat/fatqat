"""Tests statevector simulator initialization, gate application, and state export."""

import numpy as np

from fatqat.simulator.engine.np import NumpySVEngine
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.implementation.matrices import shift_matrix

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_CY = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
)
_CCX = np.eye(8, dtype=complex)
_CCX[[6, 7]] = _CCX[[7, 6]]


def _engine(n):
    eng = NumpySVEngine()
    eng.initialize((2,) * n)
    return eng


def test_initialize_zero_state():
    eng = _engine(2)
    assert np.allclose(eng.export_state(), [1, 0, 0, 0])


def test_x_on_qubit0():
    eng = _engine(1)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    assert np.allclose(eng.export_state(), [0, 1])


def test_x_on_qubit0_of_two():
    # little-endian: qubit0 is bit0, so |00> -> |01> at index 1
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    assert np.allclose(eng.export_state(), [0, 1, 0, 0])


def test_x_on_qubit1_of_two():
    # qubit1 is bit1, so |00> -> index 2
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(1,)))
    assert np.allclose(eng.export_state(), [0, 0, 1, 0])


def test_bell_state_h_then_cx():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_CX, target_indices=(0, 1)))  # control=qubit0
    assert np.allclose(eng.export_state(), [1, 0, 0, 1] / np.sqrt(2))


def test_export_state_returns_independent_copy():
    eng = _engine(1)
    first = eng.export_state()
    first[0] = 999.0
    second = eng.export_state()
    assert second[0] != 999.0


def test_swap_exchanges_two_qubits():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # qubit0=1, qubit1=0
    eng.apply(ApplyMatrixStep(matrix=_SWAP, target_indices=(0, 1)))
    assert np.allclose(eng.export_state(), [0, 0, 1, 0])  # qubit0=0, qubit1=1


def test_cy_flips_target_with_i_when_control_is_one():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # control qubit0=1
    eng.apply(ApplyMatrixStep(matrix=_CY, target_indices=(0, 1)))  # control=qubit0
    assert np.allclose(eng.export_state(), [0, 0, 0, 1j])  # qubit0=1, qubit1=1, phase i


def test_ccx_flips_target_only_when_both_controls_are_one():
    eng = _engine(3)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(1,)))
    eng.apply(
        ApplyMatrixStep(matrix=_CCX, target_indices=(0, 1, 2))
    )  # controls=qubit0,qubit1
    expected = np.zeros(8, dtype=complex)
    expected[7] = 1.0  # qubit0=1, qubit1=1, qubit2=1
    assert np.allclose(eng.export_state(), expected)


def test_ccx_leaves_target_when_one_control_is_zero():
    eng = _engine(3)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    eng.apply(
        ApplyMatrixStep(matrix=_CCX, target_indices=(0, 1, 2))
    )  # controls=qubit0,qubit1
    expected = np.zeros(8, dtype=complex)
    expected[1] = 1.0  # qubit0=1, qubit1=0, qubit2=0 (unchanged)
    assert np.allclose(eng.export_state(), expected)


def test_shift_matrix_qutrit_cycles_basis():
    s = shift_matrix(3, 1)
    # |0>->|1>, |1>->|2>, |2>->|0>
    assert np.allclose(s @ np.array([1, 0, 0], dtype=complex), [0, 1, 0])
    assert np.allclose(s @ np.array([0, 1, 0], dtype=complex), [0, 0, 1])
    assert np.allclose(s @ np.array([0, 0, 1], dtype=complex), [1, 0, 0])


def test_apply_shift_on_single_qutrit():
    eng = NumpySVEngine()
    eng.initialize((3,))
    eng.apply(ApplyMatrixStep(matrix=shift_matrix(3, 1), target_indices=(0,)))
    assert np.allclose(eng.export_state(), [0, 1, 0])  # |0> -> |1>


def test_apply_matrix_mixed_radix_qutrit_qubit():
    # 2 subsystems: dim-3 (subsystem 0) and dim-2 (subsystem 1); state size 6.
    eng = NumpySVEngine()
    eng.initialize((3, 2))
    # Shift subsystem 0 (the qutrit) by 1: |0,0> -> |1,0>.
    eng.apply(ApplyMatrixStep(matrix=shift_matrix(3, 1), target_indices=(0,)))
    # Flat index of |q0=1, q1=0> little-endian: 1 * stride0(=prod(dims[:0])=1) = 1.
    expected = np.zeros(6, dtype=complex)
    expected[1] = 1.0
    assert np.allclose(eng.export_state(), expected)
