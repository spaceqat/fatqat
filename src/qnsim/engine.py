"""Statevector engine: a stateful simulator that owns the quantum state.

The engine is the matrix family's numerical core. The backend initializes it,
feeds it resolved ``MatrixImplementation`` payloads, and reads results back
through sampling / collapse / export. The state never leaves the engine until
``export_state`` is called.

Conventions:
- little-endian: amplitude index bit ``q`` is the value of qubit ``q``.
- a ``MatrixImplementation``'s ``target_indices`` map to the matrix's local
  index with ``target_indices[0]`` as the most-significant bit.
"""

from __future__ import annotations

import numpy as np

from .implementation import MatrixImplementation


class StateVectorEngine:
    def __init__(self) -> None:
        self._state: np.ndarray | None = None
        self._n_qubits: int = 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def initialize(self, n_qubits: int) -> None:
        """Prepare the all-zero computational basis state on ``n_qubits``."""
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        self._state = state
        self._n_qubits = n_qubits

    def apply(self, impl: MatrixImplementation) -> None:
        """Evolve the state by one resolved matrix implementation in place."""
        self._require_state()
        self._state = _apply_matrix(
            self._state, impl.matrix, impl.target_indices, self._n_qubits
        )

    def probabilities(self) -> np.ndarray:
        self._require_state()
        p = np.abs(self._state) ** 2
        total = p.sum()
        return p / total if total > 0 else p

    def sample_indices(self, shots, rng) -> np.ndarray:
        self._require_state()
        return rng.choice(len(self._state), size=shots, p=self.probabilities())

    def collapse(self, measured_qubits, rng) -> dict[int, int]:
        """Sample one outcome, project the internal state, return measured bits."""
        self._require_state()
        idx = int(rng.choice(len(self._state), p=self.probabilities()))
        bits = {q: (idx >> q) & 1 for q in measured_qubits}
        arange = np.arange(len(self._state))
        keep = np.ones(len(self._state), dtype=bool)
        for q, b in bits.items():
            keep &= ((arange >> q) & 1) == b
        new = np.where(keep, self._state, 0.0).astype(complex)
        norm = np.linalg.norm(new)
        if norm > 0:
            new = new / norm
        self._state = new
        return bits

    def export_state(self) -> np.ndarray:
        self._require_state()
        return self._state

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("engine not initialized; call initialize(n_qubits) first")


def _apply_matrix(state, matrix, targets, n_qubits) -> np.ndarray:
    """Apply a 2**k matrix to flat ``targets`` of a little-endian state.

    The matrix's local index treats ``targets[0]`` as the MSB and
    ``targets[k-1]`` as the LSB.
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
