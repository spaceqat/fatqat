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

import importlib.util
from collections.abc import Callable

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


def _statevector_term_numpy(
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


def _density_matrix_term_numpy(
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


def _load_compiled_terms() -> tuple[Callable[..., complex], ...] | None:
    """Return the compiled per-term kernels, or ``None`` when numba is absent.

    Deferring the Numba import here rather than at module scope keeps package
    import lightweight and preserves the NumPy fallback for deliberately
    minimal or damaged environments.

    The absence of numba is the *only* reason this falls back. Any other import
    failure propagates: a compiled kernel that cannot load where numba is
    installed is a bug, and silently substituting the NumPy path would hide it -
    the run would still produce right answers, slowly, while the tests written
    to catch it went quiet. Checking the spec rather than catching every
    ``ImportError`` is what keeps those two cases apart.
    """
    if importlib.util.find_spec("numba") is None:
        return None
    from . import expectation_nb

    return expectation_nb.statevector_term, expectation_nb.density_matrix_term


_COMPILED = _load_compiled_terms()
USING_COMPILED_KERNEL = _COMPILED is not None


def _bind_term_evaluator(
    state: np.ndarray, compiled_index: int, fallback: Callable[..., complex]
) -> Callable[[int, int, int, int], complex]:
    """Return ``masks -> value`` for one state, with the implementation chosen.

    Choosing once per call rather than once per term is the point: the term
    loop stays a single call with no branch, and the two implementations differ
    only in what they need to close over. The NumPy form needs an index array
    across the whole state; the compiled form walks the range itself, so
    building that array for it would allocate 8 bytes per amplitude for nothing.
    """
    if _COMPILED is not None:
        kernel = _COMPILED[compiled_index]
        return lambda *masks: kernel(state, *masks)
    index = np.arange(state.shape[0])
    return lambda *masks: fallback(state, index, *masks)


def expectation_statevector(
    state: np.ndarray, terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...]
) -> float:
    """Return ``<psi|O|psi>`` for a statevector and a term list.

    The state is read, never modified or copied. Terms are evaluated against
    the same state in turn, which is the whole point of evaluating a
    many-term observable in one place: the evolution is paid for once.
    """
    term_value = _bind_term_evaluator(state, 0, _statevector_term_numpy)
    total = 0.0 + 0.0j
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue  # a zero coefficient contributes nothing; skip the pass
        x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors)
        value = term_value(x_mask, z_mask, zero_mask, one_mask)
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
    term_value = _bind_term_evaluator(rho, 1, _density_matrix_term_numpy)
    total = 0.0 + 0.0j
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue
        x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors)
        value = term_value(x_mask, z_mask, zero_mask, one_mask)
        total += coefficient * value * (1j**n_y)
    return float(total.real)
