"""Statevector engine: matrix application and measurement sampling."""

from __future__ import annotations

import numpy as np


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def apply(state, matrix, targets, n_qubits) -> np.ndarray:
    """Apply a 2**k matrix to flat `targets`.

    Conventions:
    - state is little-endian: amplitude index bit q = value of qubit q.
    - matrix local index: operand 0 is the MSB, operand k-1 the LSB.
    """
    k = len(targets)
    psi = state.reshape((2,) * n_qubits)  # axis p corresponds to qubit (n_qubits-1-p)
    target_axes = [n_qubits - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape((2,) * (2 * k))
    # m axes: [out_0..out_{k-1}, in_0..in_{k-1}]; contract inputs with target axes.
    psi = np.tensordot(m, psi, axes=(list(range(k, 2 * k)), target_axes))
    # Result axes: [out_0..out_{k-1}] + remaining state axes (original relative order).
    remaining = [ax for ax in range(n_qubits) if ax not in target_axes]
    perm = [0] * n_qubits
    for j, ax in enumerate(target_axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    psi = np.transpose(psi, perm)
    return psi.reshape(-1)
