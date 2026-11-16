"""NumPy matrix application for states and operator columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ....._backends.steps import ApplyMatrixStep


def _contract_local(
    m: np.ndarray, tensor: np.ndarray, axes: Sequence[int], total: int, k: int
) -> np.ndarray:
    """Contract a local ``[out, in]`` operator into ``axes`` of a tensor.

    ``m`` has ``2k`` axes (k out, then k in); ``tensor`` has ``total`` axes. The
    k input axes of ``m`` are contracted with ``axes`` and the resulting out axes
    are permuted back into their positions. This is the single matrix-application
    core both apply kernels share: the statevector kernel calls it once on the
    n-axis state tensor, the density-matrix kernel twice (ket then bra) on the
    2n-axis rho tensor.
    """
    out = np.tensordot(m, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    # Result axes: [out_0..out_{k-1}] + remaining tensor axes (original order).
    remaining = [ax for ax in range(total) if ax not in axes]
    perm = [0] * total
    for j, ax in enumerate(axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    return np.transpose(out, perm)


def _apply_statevector(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix using the caller's prepared subsystem dimensions."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    psi = state.reshape(reversed_dims)
    target_axes = [n - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    return _contract_local(m, psi, target_axes, n, k).reshape(-1)


def _apply_density_matrix(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Return the local sandwich M rho M-adjoint using prepared dimensions."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    tensor = rho.reshape(reversed_dims * 2)
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    tensor = _contract_local(m, tensor, ket_axes, 2 * n, k)
    tensor = _contract_local(m.conj(), tensor, bra_axes, 2 * n, k)
    return tensor.reshape(rho.shape)


def _apply_unitary(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix to every unitary column."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    columns = state.shape[1]
    tensor = state.reshape(reversed_dims + (columns,))
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    target_axes = [n - 1 - q for q in targets]
    return _contract_local(m, tensor, target_axes, n + 1, k).reshape(state.shape)


def _apply_superoperator(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix sandwich to every superoperator column."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    columns = rho.shape[1]
    tensor = rho.reshape(reversed_dims * 2 + (columns,))
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    total = 2 * n + 1
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    tensor = _contract_local(m, tensor, ket_axes, total, k)
    tensor = _contract_local(m.conj(), tensor, bra_axes, total, k)
    return tensor.reshape(rho.shape)


# --- Python-facing matrix operations ---
# pylint: disable-next=fixme
# TODO: Reuse reversed dimensions instead of rebuilding them per gate.


def _apply_matrix_statevector(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
) -> np.ndarray:
    """Apply one gate to a statevector from Python workflow operands."""
    return _apply_statevector(
        state, step.matrix, step.target_indices, dims, tuple(reversed(dims))
    )


def _apply_matrix_density_matrix(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
) -> np.ndarray:
    """Apply one gate to a density matrix from Python workflow operands."""
    return _apply_density_matrix(
        state, step.matrix, step.target_indices, dims, tuple(reversed(dims))
    )


def _apply_matrix_unitary(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
) -> np.ndarray:
    """Apply one gate to a unitary operator from Python workflow operands."""
    return _apply_unitary(
        state, step.matrix, step.target_indices, dims, tuple(reversed(dims))
    )


def _apply_matrix_superoperator(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
) -> np.ndarray:
    """Apply one gate to a superoperator from Python workflow operands."""
    return _apply_superoperator(
        state, step.matrix, step.target_indices, dims, tuple(reversed(dims))
    )
