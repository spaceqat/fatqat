"""Correctness of the in-place Numba apply-matrix prototype vs `_apply_sv`.

The Numba kernel must reproduce the reference `numpy_engine._apply_sv`
bit-for-bit (up to floating-point tolerance) across qubit, qudit, multi-target,
reversed-target, and full-system cases. Randomized cases use dense non-unitary
matrices and random complex states: the contraction is linear, so unitarity is
irrelevant to correctness and random operators stress the index bookkeeping
hardest.
"""

import numpy as np
import pytest

pytest.importorskip("numba")  # optional dependency; skip suite if absent

from fatqat.backends.numpy_engine import _apply_sv
from fatqat.backends.statevector_numba import (
    apply_matrix_inplace,
    apply_matrix_inplace_parallel,
)

# Both the serial and prange kernels must reproduce the reference identically.
_IMPLS = [apply_matrix_inplace, apply_matrix_inplace_parallel]


def _random_state(dims, rng):
    n = int(np.prod(dims))
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)


def _random_matrix(dims, targets, rng):
    dt = int(np.prod([dims[t] for t in targets]))
    return (rng.standard_normal((dt, dt)) + 1j * rng.standard_normal((dt, dt))).astype(
        np.complex128
    )


def _assert_matches_reference(state, matrix, targets, dims):
    reference = _apply_sv(state.copy(), matrix, targets, dims)
    for impl in _IMPLS:
        numba_out = state.copy()
        returned = impl(numba_out, matrix, targets, dims)
        assert returned is numba_out  # mutates in place, returns the same buffer
        assert np.allclose(numba_out, reference), impl.__name__


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def test_x_on_qubit0_of_two():
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    apply_matrix_inplace(state, _X, (0,), (2, 2))
    assert np.allclose(state, [0, 1, 0, 0])


def test_x_on_qubit1_of_two():
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    apply_matrix_inplace(state, _X, (1,), (2, 2))
    assert np.allclose(state, [0, 0, 1, 0])


def test_bell_h_then_cx():
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    apply_matrix_inplace(state, _H, (0,), (2, 2))
    apply_matrix_inplace(state, _CX, (0, 1), (2, 2))
    assert np.allclose(state, np.array([1, 0, 0, 1]) / np.sqrt(2))


@pytest.mark.parametrize(
    "dims, targets",
    [
        ((2,), (0,)),
        ((2, 2), (0,)),
        ((2, 2), (1,)),
        ((2, 2), (0, 1)),
        ((2, 2), (1, 0)),  # reversed target order: targets[0] is the local MSB
        ((2, 2, 2), (0, 2)),
        ((2, 2, 2), (2, 0)),
        ((2, 2, 2), (0, 1, 2)),  # full-system, no rest slices
        ((3,), (0,)),  # qutrit
        ((3, 2), (0,)),
        ((2, 3), (1,)),
        ((2, 3, 2), (1,)),  # mixed radix, middle qudit target
        ((2, 3, 2), (2, 0)),  # mixed radix, reversed multi-target
        ((3, 3), (0, 1)),
    ],
)
def test_matches_reference_random(dims, targets):
    rng = np.random.default_rng(20260713)
    for _ in range(5):
        state = _random_state(dims, rng)
        matrix = _random_matrix(dims, targets, rng)
        _assert_matches_reference(state, matrix, targets, dims)


def test_readonly_matrix_is_accepted():
    # ApplyMatrixStep freezes its matrix read-only; the kernel must still apply it.
    matrix = _CX.astype(np.complex128)
    matrix.flags.writeable = False
    state = np.array([0, 0, 1, 0], dtype=np.complex128)  # |10> little-endian: control=q0=0
    _assert_matches_reference(state, matrix, (0, 1), (2, 2))


def test_larger_system_random_single_and_two_qubit():
    rng = np.random.default_rng(7)
    dims = (2,) * 8
    for targets in [(0,), (7,), (3,), (0, 7), (7, 0), (2, 5)]:
        state = _random_state(dims, rng)
        matrix = _random_matrix(dims, targets, rng)
        _assert_matches_reference(state, matrix, targets, dims)
