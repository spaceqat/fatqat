"""NumPy exact expectation evaluation, and the term encoders both runtimes use.

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

The same algebra holds when subsystems have other dimensions. Paulis still act
only on qubits, and projector and number factors are diagonal on any
subsystem, so a term remains a phased permutation with a diagonal weight. Only
the index arithmetic changes: a mixed-radix digit replaces a bit, and flipping
a qubit adds or subtracts its place value instead of XOR-ing a mask. Programs
made only of qubits keep the bitmask kernels; any other dimension uses the
digit kernels.

The Numba runtime compiles the same sums in ``numba_kernels._expectation`` and
reuses the term encoders and the term loop defined here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .....observable import NumberOperator, Projector, _Factors, _Pauli

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


# (letter -> contributes to) masks. Y is both a bit flip and a sign, which is
# exactly the X*Z decomposition above.
_FLIPS = frozenset({"X", "Y"})
_SIGNS = frozenset({"Y", "Z"})


def _term_masks(
    factors: _Factors, strides: Sequence[int]
) -> tuple[int, int, int, int, int]:
    """Pack one qubit term's factors into bit masks.

    Returns ``(x_mask, z_mask, zero_mask, one_mask, n_y)``. Only the qubits the
    term actually names are visited, so this is O(factors), not O(num_qubits).
    A qubit's place value is its bit, so the masks follow whatever order the
    caller's ``strides`` describe. On a qubit the number operator is the
    ``ONE`` projector.
    """
    x_mask = z_mask = zero_mask = one_mask = n_y = 0
    for qubit, factor in factors:
        bit = strides[qubit]
        if isinstance(factor, _Pauli):
            if factor.letter in _FLIPS:
                x_mask |= bit
            if factor.letter in _SIGNS:
                z_mask |= bit
            if factor.letter == "Y":
                n_y += 1
        elif isinstance(factor, NumberOperator):
            one_mask |= bit
        elif isinstance(factor, Projector) and factor.level == 1:
            one_mask |= bit
        elif isinstance(factor, Projector) and factor.level == 0:
            zero_mask |= bit
        else:
            raise ValueError(f"factor {factor!r} has no qubit form")
    return x_mask, z_mask, zero_mask, one_mask, n_y


def _term_digits(
    factors: _Factors, dims: Sequence[int], strides: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pack one term's factors into per-factor digit-kernel arrays.

    Returns ``(strides, dims, kinds, levels, n_y)``, with each factor's stride
    and dimension taken from the caller's per-subsystem ``dims`` and ``strides``.
    """
    factor_strides = []
    factor_dims = []
    kinds = []
    levels = []
    n_y = 0
    for axis, factor in factors:
        factor_strides.append(strides[axis])
        factor_dims.append(dims[axis])
        level = 0
        if isinstance(factor, _Pauli):
            kind = {"X": KIND_X, "Y": KIND_Y, "Z": KIND_Z}[factor.letter]
            n_y += factor.letter == "Y"
        elif isinstance(factor, Projector):
            kind = KIND_PROJECTOR
            level = factor.level
        else:
            kind = KIND_NUMBER
        kinds.append(kind)
        levels.append(level)
    return (
        np.array(factor_strides, dtype=np.int64),
        np.array(factor_dims, dtype=np.int64),
        np.array(kinds, dtype=np.int64),
        np.array(levels, dtype=np.int64),
        n_y,
    )


def _sum_terms(
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
    *,
    qubit_value: Callable[[int, int, int, int], complex],
    digit_value: Callable[..., complex],
) -> float:
    """Sum one observable's terms, using the qubit or the digit kernel.

    ``qubit_value`` and ``digit_value`` evaluate a single encoded term against
    one already-bound state; each runtime supplies its own.
    """
    total = 0.0 + 0.0j
    if all(dim == 2 for dim in dims):
        for coefficient, factors in terms:
            if coefficient == 0.0:
                continue  # a zero coefficient contributes nothing; skip the pass
            x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors, strides)
            value = qubit_value(x_mask, z_mask, zero_mask, one_mask)
            total += coefficient * value * (1j**n_y)
        return float(total.real)
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue
        *arrays, n_y = _term_digits(factors, dims, strides)
        total += coefficient * digit_value(*arrays) * (1j**n_y)
    return float(total.real)


def _expectation(
    data: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
    *,
    density: bool,
) -> float:
    index = np.arange(data.shape[0])
    qubit_kernel = qubit_density_matrix_term if density else qubit_statevector_term
    return _sum_terms(
        terms,
        dims,
        strides,
        qubit_value=lambda *masks: qubit_kernel(data, index, *masks),
        digit_value=lambda *arrays: digit_term(data, index, *arrays, density=density),
    )


def expectation_statevector(
    state: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
) -> float:
    """Return ``<psi|O|psi>`` for a statevector and a term list.

    ``<psi|T|psi> = sum_k conj(psi[p(k)]) * w(k) * psi[k]``, where ``p(k)`` is
    the basis state the term's X/Y factors flip ``k`` to and ``w(k)`` is its
    diagonal weight. The state is read, never modified or copied. Terms are
    evaluated against the same state in turn, which is the whole point of
    evaluating a many-term observable in one place: the evolution is paid for
    once. ``dims`` and ``strides`` give each subsystem's dimension and place
    value in the flat basis index, indexed like the factors' subsystem numbers
    """
    return _expectation(state, terms, dims, strides, density=False)


def expectation_density_matrix(
    rho: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
) -> float:
    """Return ``Tr(rho O)`` for a density matrix and a term list.

    ``Tr(rho T) = sum_j rho[j, p(j)] * w(j)``, where ``p(j)`` is the basis
    state the term's X/Y factors flip ``j`` to and ``w(j)`` is its diagonal
    weight. The term picks out one shifted diagonal of ``rho``, so only one
    entry per row is read rather than the full square. On qubits ``p(j)`` is
    ``j XOR x_mask``. ``dims`` and ``strides`` give each subsystem's dimension and
    place value, indexed like the factors' subsystem numbers.
    """
    return _expectation(rho, terms, dims, strides, density=True)
