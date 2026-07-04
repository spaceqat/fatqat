"""Statevector engine: a stateful simulator that owns the quantum state.

Conventions:
- little-endian: flat basis index digit for subsystem ``q`` has place value
  ``prod(dims[:q])``; subsystem 0 is the least-significant digit.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from .implementation import ApplyMatrixStep, shift_matrix


class StateVectorEngine:
    """Stateful numerical core for statevector evolution.

    The engine owns the current state buffer. Backends initialize the state,
    apply resolved matrix payloads, then sample, collapse, or export copies of
    the state.
    """

    def __init__(self) -> None:
        """Create an uninitialized engine."""
        self._state: np.ndarray | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()

    @property
    def n_qubits(self) -> int:
        """Number of subsystems in the currently initialized state."""
        return len(self._dims)

    def initialize(self, dims: Sequence[int]) -> None:
        """Prepare the all-zero computational basis state over ``dims``."""
        dims = tuple(int(d) for d in dims)
        state = np.zeros(prod(dims) if dims else 1, dtype=complex)
        state[0] = 1.0
        self._state = state
        self._dims = dims
        # Cached once per circuit execution (not per gate): dims are fixed for
        # the engine's lifetime between initialize() calls, so recomputing this
        # reshape shape on every apply() would be pure per-call Python overhead.
        self._reversed_dims = tuple(reversed(dims))

    def apply(self, step: ApplyMatrixStep) -> None:
        """Evolve the state by one resolved matrix step in place."""
        self._require_state()
        self._state = _apply_matrix(
            self._state,
            step.matrix,
            step.target_indices,
            self._dims,
            self._reversed_dims,
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
        idx, new = _collapse_state(self._state, measured_qubits, self._dims, rng)
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
        return tuple(_digit(flat, index, self._dims) for index in indices)

    def measure_qubit(self, index: int, rng: np.random.Generator) -> int:
        """Sample and collapse a single qubit in the computational basis.

        Projects the internal state onto the sampled outcome for ``index`` and
        returns that qubit's measured digit. Consumes exactly one rng draw.
        """
        return self.measure_qubits((index,), rng)[0]

    def reset_qubits(self, indices: Sequence[int], rng: np.random.Generator) -> None:
        """Measure a group of qubits and reprepare them in ``|0>``."""
        self._require_state()
        if len(indices) < 1:
            raise ValueError("reset_qubits requires at least one index")
        outcomes = self.measure_qubits(indices, rng)
        for index, outcome in zip(indices, outcomes):
            if outcome != 0:
                inv = shift_matrix(self._dims[index], -outcome)
                self._state = _apply_matrix(self._state, inv, (index,), self._dims)

    def reset_qubit(self, index: int, rng: np.random.Generator) -> None:
        """Measure a qubit and reprepare it in ``|0>``.

        Samples an outcome (one rng draw), projects, and shifts the target back
        to ``|0>`` when the outcome is nonzero. The rest of an entangled state
        is left correctly conditioned on the sampled branch.
        """
        self.reset_qubits((index,), rng)

    def export_state(self) -> np.ndarray:
        """Return a copy of the current statevector."""
        self._require_state()
        return self._state.copy()

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("engine not initialized; call initialize(dims) first")


def _strides(dims: Sequence[int]) -> list[int]:
    """Little-endian place values: stride[q] = prod(dims[:q])."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def _digit(flat: int, index: int, dims: Sequence[int]) -> int:
    stride = 1
    for q in range(index):
        stride *= dims[q]
    return (flat // stride) % dims[index]


def _apply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply a local matrix to flat ``targets`` of a little-endian mixed-radix state.

    The matrix's local index treats ``targets[0]`` as the MSB and
    ``targets[k-1]`` as the LSB, sized by each target's own radix from
    ``dims``.

    ``reversed_dims`` is ``tuple(reversed(dims))``, the state's reshape shape.
    Callers that invoke this once per gate on a fixed ``dims`` (the engine's
    hot path) should precompute and pass it; direct/test callers may omit it
    and it is derived from ``dims`` at O(n) cost.

    Complexity: this is a matrix-vector contraction equivalent to an einsum
    ``M[out, in] * psi[in, rest] -> psi[out, rest]``. Work is O(prod(local_dims)
    * prod(dims)) FLOPs, and peak memory is ~2x the state: ``tensordot``
    allocates a new tensor and ``transpose`` may copy it again to reorder
    axes. An in-place variant (looping over the non-target slices and
    multiplying each local-dimension vector by M) would drop the intermediate
    allocation and improve cache locality, at the cost of more complex code.
    Deferred for now.
    """
    n = len(dims)
    k = len(targets)
    local_dims = [dims[t] for t in targets]
    if reversed_dims is None:
        reversed_dims = tuple(dims[n - 1 - p] for p in range(n))
    psi = state.reshape(tuple(reversed_dims))
    target_axes = [n - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape(tuple(local_dims) + tuple(local_dims))
    # m axes: [out_0..out_{k-1}, in_0..in_{k-1}]; contract inputs with target axes.
    psi = np.tensordot(m, psi, axes=(list(range(k, 2 * k)), target_axes))
    # Result axes: [out_0..out_{k-1}] + remaining state axes (original relative order).
    remaining = [ax for ax in range(n) if ax not in target_axes]
    perm = [0] * n
    for j, ax in enumerate(target_axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    psi = np.transpose(psi, perm)
    return psi.reshape(-1)


def _collapse_state(
    state: np.ndarray,
    measured_qubits: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Sample one computational-basis outcome and return the projected state."""
    idx = int(rng.choice(len(state), p=_probabilities(state)))
    qubits = list(measured_qubits)
    n = len(dims)

    if len(set(qubits)) == n:
        new = np.zeros_like(state)
        new[idx] = state[idx]
    else:
        strides = _strides(dims)
        basis = np.arange(len(state))
        # One (N, m) broadcast instead of a Python loop of m separate O(N)
        # passes: digits/idx_digits below fold every measured qubit's stride
        # and modulus into a single vectorized divide/mod/compare.
        stride_arr = np.array([strides[q] for q in qubits])
        dim_arr = np.array([dims[q] for q in qubits])
        digits = (basis[:, None] // stride_arr) % dim_arr
        idx_digits = (idx // stride_arr) % dim_arr
        keep = np.all(digits == idx_digits, axis=1)
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
