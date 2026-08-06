"""Expectation-value kernels for observable terms.

Computes ``<psi|O|psi>`` (statevector) and ``Tr(rho O)`` (density matrix) for
the term form produced by :py:class:`~fatqat.Observable`, without ever building
the ``2**n x 2**n`` operator.

The kernels exploit one structural fact: **every supported term is a phased
permutation with a diagonal weight.** Writing a term as its local factors,

    X, Y  move amplitude between basis states (they flip that qubit's bit)
    Z     leaves the basis state, contributing a sign
    ZERO
    ONE   leave the basis state, contributing a 0/1 weight

and using ``Y = i * X * Z``, a whole term factorizes into

    T = i**n_y * X**x_mask * Z**z_mask * (projector weights)
    T|j> = phase(j) * keep(j) * |j XOR x_mask>

with ``phase(j) = i**n_y * (-1)**popcount(j & z_mask)`` and ``keep(j)`` the 0/1
projector mask. So one pass over the amplitudes evaluates any term: an index
XOR, a popcount, and a dot product - no state copy, no per-factor matrix
application, and no dependence on how many factors the term has.

A term whose factors are all diagonal has ``x_mask == 0``, which makes the
permutation the identity; the same code path then reads as a weighted sum of
basis-state probabilities, so diagonal observables need no special case.

Every qubit carries at most one factor per term, so the projector qubits and the
X/Y qubits are disjoint. That is why the masks may be evaluated on the permuted
index without distinguishing the two groups.
"""

from __future__ import annotations

import numpy as np

# (letter -> contributes to) masks. Y is both a bit flip and a sign, which is
# exactly the X*Z decomposition above.
_FLIPS = frozenset({"X", "Y"})
_SIGNS = frozenset({"Y", "Z"})
_PROJECTORS = frozenset({"ZERO", "ONE"})


def squared_factors(
    factors: tuple[tuple[int, str], ...],
) -> tuple[tuple[int, str], ...]:
    """Return the factors of ``T**2`` for a term ``T``.

    Every Pauli squares to the identity and drops out; both projectors are
    idempotent and survive unchanged. So ``T**2`` is just the term's projector
    part, and ``<T**2>`` costs one more pass of the same kernel rather than a
    separate operator.

    Sampling needs this: a pure Pauli term has eigenvalues ``+-1`` and so
    ``<T**2> = 1``, but a term carrying a projector has eigenvalues ``{0, +-1}``
    and its second moment is a property of the state.
    """
    return tuple((qubit, letter) for qubit, letter in factors if letter in _PROJECTORS)


def _term_masks(factors: tuple[tuple[int, str], ...]) -> tuple[int, int, int, int, int]:
    """Pack one term's factors into bit masks.

    Returns ``(x_mask, z_mask, zero_mask, one_mask, n_y)``. Only the qubits the
    term actually names are visited, so this is O(factors), not O(num_qubits).
    """
    x_mask = z_mask = zero_mask = one_mask = n_y = 0
    for qubit, letter in factors:
        bit = 1 << qubit
        if letter in _FLIPS:
            x_mask |= bit
        if letter in _SIGNS:
            z_mask |= bit
        if letter == "Y":
            n_y += 1
        elif letter == "ZERO":
            zero_mask |= bit
        elif letter == "ONE":
            one_mask |= bit
    return x_mask, z_mask, zero_mask, one_mask, n_y


def _weights(
    index: np.ndarray, z_mask: int, zero_mask: int, one_mask: int
) -> np.ndarray:
    """Per-basis-state weight from a term's diagonal factors.

    Combines the Z sign ``(-1)**popcount(index & z_mask)`` with the projector
    mask, which keeps only states whose bits match what the projectors select.

    ``np.bitwise_count`` returns ``uint8``; computing the sign by arithmetic on
    it (``1 - 2 * count``) would wrap around to 255 instead of -1, so the sign
    is selected rather than computed.
    """
    weight = np.where(np.bitwise_count(index & z_mask) & 1, -1.0, 1.0)
    if zero_mask or one_mask:
        keep = ((index & one_mask) == one_mask) & ((index & zero_mask) == 0)
        weight = weight * keep
    return weight


def expectation_statevector(
    state: np.ndarray, terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...]
) -> float:
    """Return ``<psi|O|psi>`` for a statevector and a term list.

    The state is read, never modified or copied. Terms are evaluated against
    the same state in turn, which is the whole point of evaluating a
    many-term observable in one place: the evolution is paid for once.
    """
    index = np.arange(state.shape[0])
    total = 0.0 + 0.0j
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue  # a zero coefficient contributes nothing; skip the pass
        x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors)
        permuted = index ^ x_mask
        weight = _weights(permuted, z_mask, zero_mask, one_mask)
        # <psi|T|psi> = sum_k conj(psi_k) * weight(k^x) * psi_{k^x}
        value = np.vdot(state, weight * state[permuted])
        total += coefficient * value * (1j**n_y)
    return float(total.real)


def expectation_density_matrix(
    rho: np.ndarray, terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...]
) -> float:
    """Return ``Tr(rho O)`` for a density matrix and a term list.

    ``Tr(rho T) = sum_j rho[j, j XOR x_mask] * phase(j) * keep(j)`` - the term
    picks out one shifted diagonal of ``rho``, so only ``2**n`` entries are
    read per term rather than the full ``4**n`` matrix.
    """
    index = np.arange(rho.shape[0])
    total = 0.0 + 0.0j
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue
        x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors)
        weight = _weights(index, z_mask, zero_mask, one_mask)
        shifted_diagonal = rho[index, index ^ x_mask]
        total += coefficient * np.sum(weight * shifted_diagonal) * (1j**n_y)
    return float(total.real)
