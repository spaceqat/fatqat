"""Statevector engine: a stateful simulator that owns the quantum state.

The engine is the matrix family's numerical core. The backend initializes it,
feeds it resolved ``ApplyMatrixStep`` payloads, and reads results back through
sampling / collapse / export. The state never leaves the engine until
``export_state`` is called.

Conventions:
- little-endian: amplitude index bit ``q`` is the value of qubit ``q``.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant bit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .implementation import ApplyMatrixStep

# Single-qubit X used to flip a measured qubit back to |0> during reset.
_X = np.array([[0, 1], [1, 0]], dtype=complex)


class StateVectorEngine:
    """Stateful numerical core for statevector evolution.

    The engine owns the current state buffer. Backends initialize the state,
    apply resolved matrix payloads, then sample, collapse, or export copies of
    the state.
    """

    def __init__(self) -> None:
        """Create an uninitialized engine."""
        self._state: np.ndarray | None = None
        self._n_qubits: int = 0

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the currently initialized state."""
        return self._n_qubits

    def initialize(self, n_qubits: int) -> None:
        """Prepare the all-zero computational basis state on ``n_qubits``."""
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        self._state = state
        self._n_qubits = n_qubits

    def apply(self, step: ApplyMatrixStep) -> None:
        """Evolve the state by one resolved matrix step in place."""
        self._require_state()
        self._state = _apply_matrix(
            self._state, step.matrix, step.target_indices, self._n_qubits
        )

    def probabilities(self) -> np.ndarray:
        """Return normalized computational-basis probabilities."""
        self._require_state()
        return _probabilities(self._state)

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample flat basis-state indices from the current state.

        Args:
            shots: Number of samples to draw.
            rng: NumPy random generator used for sampling.

        Returns:
            One-dimensional array of sampled flat basis-state indices.
        """
        self._require_state()
        return rng.choice(len(self._state), size=shots, p=self.probabilities())

    def collapse(self, measured_qubits: Sequence[int], rng: np.random.Generator) -> int:
        """Sample one outcome, project the internal state, return the flat index."""
        self._require_state()
        idx, new = _collapse_state(self._state, measured_qubits, rng)
        self._state = new
        return idx

    def measure_qubits(
        self,
        indices: Sequence[int],
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        """Sample and collapse a group of qubits in one computational-basis event."""
        self._require_state()
        if len(indices) < 1:
            raise ValueError("measure_qubits requires at least one index")
        flat = self.collapse(indices, rng)
        return tuple((flat >> index) & 1 for index in indices)

    def measure_qubit(self, index: int, rng: np.random.Generator) -> int:
        """Sample and collapse a single qubit in the computational basis.

        Projects the internal state onto the sampled outcome for ``index`` and
        returns that qubit's measured bit. Consumes exactly one rng draw.
        """
        return self.measure_qubits((index,), rng)[0]

    def reset_qubits(self, indices: Sequence[int], rng: np.random.Generator) -> None:
        """Measure a group of qubits and reprepare them in ``|0>``."""
        self._require_state()
        if len(indices) < 1:
            raise ValueError("reset_qubits requires at least one index")
        bits = self.measure_qubits(indices, rng)
        for index, bit in zip(indices, bits):
            if bit == 1:
                self._state = _apply_matrix(self._state, _X, (index,), self._n_qubits)

    def reset_qubit(self, index: int, rng: np.random.Generator) -> None:
        """Measure a qubit and reprepare it in ``|0>``.

        Samples an outcome (one rng draw), projects, and flips the target with X
        when the outcome is 1. The rest of an entangled state is left correctly
        conditioned on the sampled branch.
        """
        self.reset_qubits((index,), rng)

    def export_state(self) -> np.ndarray:
        """Return a copy of the current statevector."""
        self._require_state()
        return self._state.copy()

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("engine not initialized; call initialize(n_qubits) first")


def _apply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """Apply a 2**k matrix to flat ``targets`` of a little-endian state.

    The matrix's local index treats ``targets[0]`` as the MSB and
    ``targets[k-1]`` as the LSB.

    Complexity: this is a matrix-vector contraction equivalent to an einsum
    ``M[out, in] * psi[in, rest] -> psi[out, rest]``. Work is O(2**k * 2**n)
    FLOPs (O(2**n) for fixed small k), and peak memory is ~2x the state:
    ``tensordot`` allocates a new O(2**n) tensor and ``transpose`` may copy it
    again to reorder axes. An in-place variant (looping over the 2**(n-k)
    non-target slices and multiplying each 2**k vector by M) would drop the
    intermediate allocation and improve cache locality, at the cost of more
    complex code. Deferred for now.
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


def _collapse_state(
    state: np.ndarray,
    measured_qubits: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Sample one computational-basis outcome and return the projected state."""
    idx = int(rng.choice(len(state), p=_probabilities(state)))
    qubits = np.asarray(measured_qubits, dtype=np.uintp)
    n_qubits = int(np.log2(len(state)))

    if qubits.size == n_qubits and np.unique(qubits).size == n_qubits:
        new = np.zeros_like(state)
        new[idx] = state[idx]
    else:
        measured_mask = (
            np.uintp(0)
            if qubits.size == 0
            else np.bitwise_or.reduce(np.left_shift(np.uintp(1), qubits))
        )
        basis = np.arange(len(state), dtype=np.uintp)
        keep = ((basis ^ np.uintp(idx)) & measured_mask) == 0
        new = state.copy()
        new[~keep] = 0.0

    norm = np.linalg.norm(new)
    if norm > 0:
        new = new / norm
    return idx, new


def _probabilities(state: np.ndarray) -> np.ndarray:
    """Return normalized computational-basis probabilities for a statevector."""
    probabilities = np.abs(state) ** 2
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities
