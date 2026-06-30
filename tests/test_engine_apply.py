import numpy as np

from qnsim.engine import StateVectorEngine
from qnsim.implementation import MatrixImplementation


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _engine(n):
    eng = StateVectorEngine()
    eng.initialize(n)
    return eng


def test_initialize_zero_state():
    eng = _engine(2)
    assert np.allclose(eng.export_state(), [1, 0, 0, 0])


def test_x_on_qubit0():
    eng = _engine(1)
    eng.apply(MatrixImplementation(matrix=_X, target_indices=(0,)))
    assert np.allclose(eng.export_state(), [0, 1])


def test_x_on_qubit0_of_two():
    # little-endian: qubit0 is bit0, so |00> -> |01> at index 1
    eng = _engine(2)
    eng.apply(MatrixImplementation(matrix=_X, target_indices=(0,)))
    assert np.allclose(eng.export_state(), [0, 1, 0, 0])


def test_x_on_qubit1_of_two():
    # qubit1 is bit1, so |00> -> index 2
    eng = _engine(2)
    eng.apply(MatrixImplementation(matrix=_X, target_indices=(1,)))
    assert np.allclose(eng.export_state(), [0, 0, 1, 0])


def test_bell_state_h_then_cx():
    eng = _engine(2)
    eng.apply(MatrixImplementation(matrix=_H, target_indices=(0,)))
    eng.apply(MatrixImplementation(matrix=_CX, target_indices=(0, 1)))  # control=qubit0
    assert np.allclose(eng.export_state(), [1, 0, 0, 1] / np.sqrt(2))
