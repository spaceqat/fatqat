import numpy as np

from qnsim.engine import zero_state, apply


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def test_zero_state():
    assert np.allclose(zero_state(2), [1, 0, 0, 0])


def test_x_on_qubit0():
    s = apply(zero_state(1), _X, (0,), 1)
    assert np.allclose(s, [0, 1])


def test_x_on_qubit0_of_two():
    # little-endian: qubit0 is bit0, so |00> -> |01> at index 1
    s = apply(zero_state(2), _X, (0,), 2)
    assert np.allclose(s, [0, 1, 0, 0])


def test_x_on_qubit1_of_two():
    # qubit1 is bit1, so |00> -> index 2
    s = apply(zero_state(2), _X, (1,), 2)
    assert np.allclose(s, [0, 0, 1, 0])


def test_bell_state_h_then_cx():
    s = zero_state(2)
    s = apply(s, _H, (0,), 2)
    s = apply(s, _CX, (0, 1), 2)  # control=qubit0, target=qubit1
    assert np.allclose(s, [1, 0, 0, 1] / np.sqrt(2))
