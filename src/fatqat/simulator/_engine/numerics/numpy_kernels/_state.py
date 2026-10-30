"""NumPy state and operator buffer allocation."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ...state import _LoweredBasisState


def _basis_index(levels: Sequence[int], dims: Sequence[int]) -> int:
    """Return the flat index of engine-order levels; subsystem 0 varies fastest."""
    index = 0
    for level, dim in zip(reversed(levels), reversed(dims), strict=True):
        index = index * dim + level
    return index


def _allocate_statevector(
    dims: Sequence[int], start: np.ndarray | _LoweredBasisState
) -> np.ndarray:
    """Return an owned ket, built as a basis state or copied from an array."""
    if isinstance(start, _LoweredBasisState):
        state = np.zeros(prod(dims), dtype=complex)
        state[_basis_index(start.levels, dims)] = 1.0
        return state
    # Copy: each shot evolves its own buffer, and the caller's array must
    # survive the run unchanged.
    return np.array(start, dtype=complex).reshape(prod(dims))


def _allocate_density_matrix(
    dims: Sequence[int], start: np.ndarray | _LoweredBasisState
) -> np.ndarray:
    """Return an owned density matrix, accepting a ket as a pure state."""
    if isinstance(start, _LoweredBasisState):
        size, index = prod(dims), _basis_index(start.levels, dims)
        state = np.zeros((size, size), dtype=complex)
        state[index, index] = 1.0
        return state
    given = np.asarray(start, dtype=complex)
    # A ket is accepted as shorthand for the pure state it describes, so the
    # same array can start a statevector and a density-matrix run without
    # the caller forming the outer product twice.
    if given.ndim == 1:
        return np.outer(given, given.conj())
    return np.array(given, dtype=complex)


def _allocate_unitary(dims: Sequence[int], start: None) -> np.ndarray:
    """Return the identity map over statevector columns."""
    assert start is None, "operator execution has no initial state"
    return np.eye(prod(dims), dtype=complex)


def _allocate_superoperator(dims: Sequence[int], start: None) -> np.ndarray:
    """Return the identity map over density-matrix columns."""
    assert start is None, "operator execution has no initial state"
    return np.eye(prod(dims) ** 2, dtype=complex)


def _export_superoperator(data: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Convert row-stacked superoperator data to public column stacking.

    This vectorization conversion does not reorder subsystems.
    """
    size = prod(dims) if dims else 1
    internal = data.reshape((size,) * 4)
    exported = np.empty_like(data, order="C")
    exported.reshape((size,) * 4)[...] = internal.transpose(1, 0, 3, 2)
    return exported
