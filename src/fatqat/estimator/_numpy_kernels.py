"""NumPy term kernels for the expectation-value pass.

The array-operation form of the arithmetic that :mod:`._numba_kernels` compiles
(see :mod:`._exact` for why a term is a phased permutation): the same
sums, expressed as whole-array operations over an index array. They are the
fallback when Numba is absent and the reference the compiled kernels are tested
against.
"""

from __future__ import annotations

import numpy as np

# Digit-kernel factor kinds. Y flips like X and signs like Z.
KIND_X, KIND_Y, KIND_Z, KIND_PROJECTOR, KIND_NUMBER = range(5)


def _weights(
    index: np.ndarray, z_mask: int, zero_mask: int, one_mask: int
) -> np.ndarray:
    """Per-basis-state weight from a term's diagonal factors.

    Combines the Z sign ``(-1)**popcount(index & z_mask)`` with the projector
    mask, which keeps only states whose bits match what the projectors select.

    XOR-folding computes parity without requiring ``np.bitwise_count``, which
    was added after the oldest NumPy supported by Numba 0.59. The sign is then
    selected rather than computed arithmetically on an unsigned value.
    """
    parity = np.asarray(index & z_mask, dtype=np.uint64)
    parity = parity ^ (parity >> 32)
    parity = parity ^ (parity >> 16)
    parity = parity ^ (parity >> 8)
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    weight = np.where(parity & 1, -1.0, 1.0)
    if zero_mask or one_mask:
        keep = ((index & one_mask) == one_mask) & ((index & zero_mask) == 0)
        weight = weight * keep
    return weight


def qubit_statevector_term(
    state: np.ndarray,
    index: np.ndarray,
    x_mask: int,
    z_mask: int,
    zero_mask: int,
    one_mask: int,
) -> complex:
    """Return ``<psi|T|psi>`` for one term, as array operations."""
    permuted = index ^ x_mask
    weight = _weights(permuted, z_mask, zero_mask, one_mask)
    # <psi|T|psi> = sum_k conj(psi_k) * weight(k^x) * psi_{k^x}
    return complex(np.vdot(state, weight * state[permuted]))


def qubit_density_matrix_term(
    rho: np.ndarray,
    index: np.ndarray,
    x_mask: int,
    z_mask: int,
    zero_mask: int,
    one_mask: int,
) -> complex:
    """Return ``Tr(rho T)`` for one term, as array operations."""
    weight = _weights(index, z_mask, zero_mask, one_mask)
    shifted_diagonal = rho[index, index ^ x_mask]
    return complex(np.sum(weight * shifted_diagonal))


def digit_term(
    data: np.ndarray,
    index: np.ndarray,
    strides: np.ndarray,
    dims: np.ndarray,
    kinds: np.ndarray,
    levels: np.ndarray,
    *,
    density: bool,
) -> complex:
    """Return one term's value on a mixed-dimension state, as array operations.

    A term maps basis state ``j`` to ``w(j) |p(j)>``, where ``p(j)`` flips the
    X/Y qubits and ``w(j)`` collects the diagonal factors, read on ``j``. So
    ``<psi|T|psi> = sum_j conj(psi_{p(j)}) w(j) psi_j`` and
    ``Tr(rho T) = sum_j rho[j, p(j)] w(j)`` share one weight.
    """
    partner = index.copy()
    weight = np.ones(index.shape[0])
    for stride, dim, kind, level in zip(strides, dims, kinds, levels):
        digit = (index // stride) % dim
        if kind in (KIND_X, KIND_Y):
            partner += stride * (1 - 2 * digit)
        if kind in (KIND_Y, KIND_Z):
            weight = weight * np.where(digit == 1, -1.0, 1.0)
        elif kind == KIND_PROJECTOR:
            weight = weight * (digit == level)
        elif kind == KIND_NUMBER:
            weight = weight * digit
    if density:
        return complex(np.sum(weight * data[index, partner]))
    return complex(np.vdot(data[partner], weight * data))
