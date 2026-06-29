"""Statevector engine: matrix application and measurement sampling."""

from __future__ import annotations

import numpy as np

# TODO: This zero_state should probably be part of the backend method and passed to the engine, rather than being part of the engine itself. The engine should be agnostic to how the state is initialized.
def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state

# TODO: This name should be more specific, because we may have other ways of applying matrices, and should be part of this particular engine, not a general function
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


def probabilities(state) -> np.ndarray:
    p = np.abs(state) ** 2
    total = p.sum()
    return p / total if total > 0 else p


def sample_indices(state, shots, rng) -> np.ndarray:
    p = probabilities(state)
    return rng.choice(len(state), size=shots, p=p)


def collapse(state, n_qubits, measured_qubits, rng):
    p = probabilities(state)
    idx = int(rng.choice(len(state), p=p))
    bits = {q: (idx >> q) & 1 for q in measured_qubits}
    arange = np.arange(len(state))
    keep = np.ones(len(state), dtype=bool)
    for q, b in bits.items():
        keep &= (((arange >> q) & 1) == b)
    new = np.where(keep, state, 0.0).astype(complex)
    norm = np.linalg.norm(new)
    if norm > 0:
        new = new / norm
    return new, bits
