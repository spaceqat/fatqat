"""Numba inner kernels for the expectation-value pass.

Same arithmetic as the NumPy kernels in :mod:`.expectation`, same term algebra
(see that module for why a term is a phased permutation) - only the innermost
sweep over ``2**n`` amplitudes moves into compiled code. The term loop, the mask
packing, and the coefficient bookkeeping stay in Python, where they cost
O(factors) and never touch the state.

The win is not "NumPy is slow". It is that the NumPy form has to materialize
what a loop can consume in flight: per term it allocates the permuted index, the
popcount, the sign, the projector mask, and the gathered ``state[permuted]`` -
five full-length temporaries, one of them a fancy-index gather that reads memory
in permuted order. The compiled loop keeps the running sum in a register and
touches each amplitude once.

This module imports ``numba`` at module scope and is imported lazily, so a
fatqat installed without the optional ``numba`` group simply never loads it.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _odd_parity(value: int) -> bool:
    """Whether ``value`` has an odd number of set bits.

    The kernels only ever need ``popcount(...) & 1``, and parity folds in six
    XOR-shifts without counting anything. Written out rather than called from
    NumPy because ``np.bitwise_count`` is not available in nopython mode - and
    would return ``uint8``, the trap the NumPy kernel documents.
    """
    folded = value
    folded ^= folded >> 32
    folded ^= folded >> 16
    folded ^= folded >> 8
    folded ^= folded >> 4
    folded ^= folded >> 2
    folded ^= folded >> 1
    return (folded & 1) == 1


# Summation block length. A flat running sum accumulates rounding error as
# O(N * eps), which at N = 2**20 is a real loss next to what `np.vdot` gives:
# NumPy sums pairwise, growing as O(log N * eps). Summing per block and then
# summing the blocks bounds the growth by O((N/B + B) * eps), which at B = 1024
# is ~2e3 instead of ~1e6 - close enough to pairwise that the compiled kernel
# is not the less accurate option. It also gives the inner loop an accumulator
# the compiler can keep in a register across the block.
_BLOCK = 1024


@njit(cache=True)
def statevector_term(
    state: np.ndarray,
    x_mask: int,
    z_mask: int,
    zero_mask: int,
    one_mask: int,
) -> complex:
    """Return ``<psi|T|psi>`` for one term, given its masks.

    Mirrors the NumPy form, including which index the diagonal masks are
    evaluated on: ``sum_i conj(psi_i) * weight(i XOR x) * psi_{i XOR x}``.
    """
    size = state.shape[0]
    total = 0.0 + 0.0j
    for start in range(0, size, _BLOCK):
        stop = min(start + _BLOCK, size)
        block = 0.0 + 0.0j
        for i in range(start, stop):
            permuted = i ^ x_mask
            if (permuted & one_mask) != one_mask or (permuted & zero_mask) != 0:
                continue  # a projector rejected this basis state
            contribution = np.conj(state[i]) * state[permuted]
            if _odd_parity(permuted & z_mask):
                block -= contribution
            else:
                block += contribution
        total += block
    return total


@njit(cache=True)
def density_matrix_term(
    rho: np.ndarray,
    x_mask: int,
    z_mask: int,
    zero_mask: int,
    one_mask: int,
) -> complex:
    """Return ``Tr(rho T)`` for one term, given its masks.

    The term selects one shifted diagonal, ``rho[j, j XOR x_mask]``, so this
    reads ``2**n`` entries rather than the full ``4**n`` matrix - the same
    access pattern as the NumPy form, without gathering it into a temporary.
    """
    size = rho.shape[0]
    total = 0.0 + 0.0j
    for start in range(0, size, _BLOCK):
        stop = min(start + _BLOCK, size)
        block = 0.0 + 0.0j
        for j in range(start, stop):
            if (j & one_mask) != one_mask or (j & zero_mask) != 0:
                continue
            entry = rho[j, j ^ x_mask]
            if _odd_parity(j & z_mask):
                block -= entry
            else:
                block += entry
        total += block
    return total
