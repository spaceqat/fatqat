"""Numba inner kernels for the expectation-value pass.

Same arithmetic as the NumPy kernels in ``numpy_kernels._expectation``, same term
algebra (see that module for why a term is a phased permutation) - only the innermost
sweep over ``2**n`` amplitudes moves into compiled code. The term loop, the mask
packing, and the coefficient bookkeeping stay in Python, where they cost
O(factors) and never touch the state.

The caller's per-subsystem place values are translated into the small bit
masks (or passed as strides) before these kernels run, so the compiled loops
never need to know which subsystem is most significant and never permute or
copy the state.

The win is not "NumPy is slow". It is that the NumPy form has to materialize
what a loop can consume in flight: per term it allocates the permuted index, the
popcount, the sign, the projector mask, and the gathered ``state[permuted]`` -
five full-length temporaries, one of them a fancy-index gather that reads memory
in permuted order. The compiled loop keeps the running sum in a register and
touches each amplitude once.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

from .....observable import _Factors
from ..numpy_kernels._expectation import _layout, _sum_terms, _table, _term_masks
from ._apply import _advance_base, _contiguous_run, _spread_base


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

# The mixed-dimension kernels flush their running sum after this many odometer
# steps; the inner sweeps along a contiguous run are blocked by `_BLOCK`.
_BLOCK_STEPS = 256


@njit(cache=True)
def qubit_statevector_term(
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


# Largest digit table the mixed-dimension kernels build. Rows are 24 bytes, so
# this caps the table at 1.5 MiB. A term that names more combinations than this
# (many Paulis or number operators) leaves few subsystems to sweep, so the
# table would save little and could grow as large as the state itself; such a
# term is evaluated by the streaming kernels instead.
TABLE_ROWS = 1 << 16


@njit(cache=True)
def _digit_partner_and_weight(
    j: int,
    strides: np.ndarray,
    dims: np.ndarray,
    kinds: np.ndarray,
    levels: np.ndarray,
) -> tuple[int, float]:
    """Return basis state ``j``'s flip partner and the term's weight on ``j``.

    Kinds 0, 1, 2 are X, Y, Z (on qubits), 3 a projector onto ``levels[f]``,
    and 4 the number operator.
    """
    partner = j
    weight = 1.0
    for f in range(kinds.shape[0]):
        digit = (j // strides[f]) % dims[f]
        kind = kinds[f]
        if kind <= 1:
            partner += strides[f] * (1 - 2 * digit)
        if kind in (1, 2):
            if digit == 1:
                weight = -weight
        elif kind == 3:
            if digit != levels[f]:
                return partner, 0.0
        elif kind == 4:
            weight *= digit
    return partner, weight


@njit(cache=True)
def streaming_statevector_term(
    state: np.ndarray,
    strides: np.ndarray,
    dims: np.ndarray,
    kinds: np.ndarray,
    levels: np.ndarray,
) -> complex:
    """Return ``<psi|T|psi>`` decoding each basis state's digits in turn.

    The fallback for a term whose table would exceed :data:`TABLE_ROWS`. It
    needs no memory beyond the state.
    """
    size = state.shape[0]
    total = 0.0 + 0.0j
    for start in range(0, size, _BLOCK):
        stop = min(start + _BLOCK, size)
        block = 0.0 + 0.0j
        for k in range(start, stop):
            partner, weight = _digit_partner_and_weight(k, strides, dims, kinds, levels)
            if weight != 0.0:
                block += weight * np.conj(state[partner]) * state[k]
        total += block
    return total


@njit(cache=True)
def streaming_density_matrix_term(
    rho: np.ndarray,
    strides: np.ndarray,
    dims: np.ndarray,
    kinds: np.ndarray,
    levels: np.ndarray,
) -> complex:
    """Return ``Tr(rho T)`` decoding each basis state's digits in turn."""
    size = rho.shape[0]
    total = 0.0 + 0.0j
    for start in range(0, size, _BLOCK):
        stop = min(start + _BLOCK, size)
        block = 0.0 + 0.0j
        for j in range(start, stop):
            partner, weight = _digit_partner_and_weight(j, strides, dims, kinds, levels)
            if weight != 0.0:
                block += weight * rho[j, partner]
        total += block
    return total


@njit(cache=True)
def statevector_term(
    state: np.ndarray,
    offsets: np.ndarray,
    partners: np.ndarray,
    weights: np.ndarray,
    rest_strides: np.ndarray,
    rest_dims: np.ndarray,
) -> complex:
    """Return ``<psi|T|psi>`` for one term on a mixed-dimension state.

    Mirrors ``table_term`` in the NumPy module, and reads the same table:
    ``sum_k conj(psi_{p(k)}) * weight(k) * psi_k``, one row of ``offsets``,
    ``partners`` and ``weights`` per combination of the named digits. The
    remaining subsystems, ``rest_strides`` and ``rest_dims`` in ascending stride
    order, are walked with the coset odometer, and the leading run of them that
    is contiguous in memory is swept as an inner loop.
    """
    run_count, run_length = _contiguous_run(rest_strides, rest_dims)
    outer_strides = rest_strides[run_count:]
    outer_dims = rest_dims[run_count:]
    n_outer = 1
    for r in range(outer_dims.shape[0]):
        n_outer *= outer_dims[r]
    counter = np.empty(outer_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, outer_strides, outer_dims, counter)
    total = 0.0 + 0.0j
    block = 0.0 + 0.0j
    for step in range(n_outer):
        for row in range(offsets.shape[0]):
            ket = base + offsets[row]
            bra = base + partners[row]
            for first in range(0, run_length, _BLOCK):
                partial = 0.0 + 0.0j
                for k in range(first, min(first + _BLOCK, run_length)):
                    partial += np.conj(state[bra + k]) * state[ket + k]
                block += weights[row] * partial
        if (step + 1) % _BLOCK_STEPS == 0:
            total += block
            block = 0.0 + 0.0j
        base = _advance_base(base, outer_strides, outer_dims, counter)
    return total + block


@njit(cache=True)
def density_matrix_term(
    rho: np.ndarray,
    offsets: np.ndarray,
    partners: np.ndarray,
    weights: np.ndarray,
    rest_strides: np.ndarray,
    rest_dims: np.ndarray,
) -> complex:
    """Return ``Tr(rho T)`` for one term on a mixed-dimension state.

    Mirrors ``table_term`` in the NumPy module: ``sum_j rho[j, p(j)] * weight(j)``,
    walked as in :func:`statevector_term`.
    """
    run_count, run_length = _contiguous_run(rest_strides, rest_dims)
    outer_strides = rest_strides[run_count:]
    outer_dims = rest_dims[run_count:]
    n_outer = 1
    for r in range(outer_dims.shape[0]):
        n_outer *= outer_dims[r]
    counter = np.empty(outer_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, outer_strides, outer_dims, counter)
    total = 0.0 + 0.0j
    block = 0.0 + 0.0j
    for step in range(n_outer):
        for row in range(offsets.shape[0]):
            ket = base + offsets[row]
            bra = base + partners[row]
            for first in range(0, run_length, _BLOCK):
                partial = 0.0 + 0.0j
                for k in range(first, min(first + _BLOCK, run_length)):
                    partial += rho[ket + k, bra + k]
                block += weights[row] * partial
        if (step + 1) % _BLOCK_STEPS == 0:
            total += block
            block = 0.0 + 0.0j
        base = _advance_base(base, outer_strides, outer_dims, counter)
    return total + block


@njit(cache=True)
def qubit_density_matrix_term(
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


def _expectation(
    data: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
    *,
    density: bool,
) -> float:
    qubits = all(dim == 2 for dim in dims)
    qubit_kernel = qubit_density_matrix_term if density else qubit_statevector_term
    table_kernel = density_matrix_term if density else statevector_term
    streaming_kernel = (
        streaming_density_matrix_term if density else streaming_statevector_term
    )

    def term_value(factors: _Factors) -> complex:
        if qubits:
            return qubit_kernel(data, *_term_masks(factors, strides))
        term = _layout(factors, dims, strides)
        if term.rows > TABLE_ROWS:
            return streaming_kernel(
                data, term.strides, term.dims, term.kinds, term.levels
            )
        return table_kernel(data, *_table(term), term.rest_strides, term.rest_dims)

    return _sum_terms(terms, term_value)


def expectation_statevector(
    state: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
) -> float:
    """Return ``<psi|O|psi>`` with the compiled kernels (see the NumPy form)."""
    return _expectation(state, terms, dims, strides, density=False)


def expectation_density_matrix(
    rho: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
) -> float:
    """Return ``Tr(rho O)`` with the compiled kernels (see the NumPy form)."""
    return _expectation(rho, terms, dims, strides, density=True)
