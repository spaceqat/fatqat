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

A term reads only the digits of the subsystems it names, so it can also be
evaluated a combination of those digits at a time. :func:`_layout` lists the
combinations a term keeps - a projector keeps one digit, so it is not
enumerated - and the subsystems it does not name. :func:`table_term` then reads
each combination as one strided view of the state over those other
subsystems, without the full-length index arrays the per-amplitude kernels
build. That is the faster form whenever the table is short next to the state;
the per-amplitude kernels remain for terms that name many subsystems.

The Numba runtime compiles the same sums in ``numba_kernels._expectation`` and
reuses the term encoders and the term loop defined here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple

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


def _term_masks(factors: _Factors, strides: Sequence[int]) -> tuple[int, int, int, int]:
    """Pack one qubit term's factors into bit masks.

    Returns ``(x_mask, z_mask, zero_mask, one_mask)``. Only the qubits the term
    actually names are visited, so this is O(factors), not O(num_qubits). A
    qubit's place value is its bit, so the masks follow whatever order the
    caller's ``strides`` describe. On a qubit the number operator is the
    ``ONE`` projector.
    """
    x_mask = z_mask = zero_mask = one_mask = 0
    for qubit, factor in factors:
        bit = strides[qubit]
        if isinstance(factor, _Pauli):
            if factor.letter in _FLIPS:
                x_mask |= bit
            if factor.letter in _SIGNS:
                z_mask |= bit
        elif isinstance(factor, NumberOperator):
            one_mask |= bit
        elif isinstance(factor, Projector) and factor.level == 1:
            one_mask |= bit
        elif isinstance(factor, Projector) and factor.level == 0:
            zero_mask |= bit
        else:
            raise ValueError(f"factor {factor!r} has no qubit form")
    return x_mask, z_mask, zero_mask, one_mask


class _Layout(NamedTuple):
    """One term as the digit and table kernels read it.

    ``strides``, ``dims``, ``kinds`` and ``levels`` describe the named
    subsystems, one entry per factor. ``rest_strides`` and ``rest_dims`` are
    the subsystems the term does not name, in ascending stride order, with
    each run of them that is contiguous in the flat index merged into one.
    ``rows`` counts the digit combinations the term's table enumerates.
    """

    strides: np.ndarray
    dims: np.ndarray
    kinds: np.ndarray
    levels: np.ndarray
    rest_strides: np.ndarray
    rest_dims: np.ndarray
    rows: int


def _layout(factors: _Factors, dims: Sequence[int], strides: Sequence[int]) -> _Layout:
    """Encode one term for the digit and table kernels.

    Each factor's stride and dimension come from the caller's per-subsystem
    ``dims`` and ``strides``. A projector admits a single digit, so it does not
    multiply ``rows``.
    """
    factor_strides = []
    factor_dims = []
    kinds = []
    levels = []
    rows = 1
    for axis, factor in factors:
        factor_strides.append(strides[axis])
        factor_dims.append(dims[axis])
        level = 0
        if isinstance(factor, _Pauli):
            kind = {"X": KIND_X, "Y": KIND_Y, "Z": KIND_Z}[factor.letter]
        elif isinstance(factor, Projector):
            kind = KIND_PROJECTOR
            level = factor.level
        else:
            kind = KIND_NUMBER
        if kind != KIND_PROJECTOR:
            rows *= dims[axis]
        kinds.append(kind)
        levels.append(level)
    named = {axis for axis, _factor in factors}
    rest: list[list[int]] = []
    for stride, dim in sorted(
        (strides[axis], dims[axis]) for axis in range(len(dims)) if axis not in named
    ):
        if rest and rest[-1][0] * rest[-1][1] == stride:
            rest[-1][1] *= dim  # continues the previous run
        else:
            rest.append([stride, dim])
    return _Layout(
        np.array(factor_strides, dtype=np.int64),
        np.array(factor_dims, dtype=np.int64),
        np.array(kinds, dtype=np.int64),
        np.array(levels, dtype=np.int64),
        np.array([stride for stride, _dim in rest], dtype=np.int64),
        np.array([dim for _stride, dim in rest], dtype=np.int64),
        rows,
    )


def _table(term: _Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a term's ``(offsets, partners, weights)`` with zero weights dropped.

    One row per combination of the named digits: the flat offset of the
    combination, the offset of its flip partner, and the term's weight there.
    Projectors are fixed at their level, so they contribute to every offset
    and never add rows.
    """
    fixed = term.kinds == KIND_PROJECTOR
    base = int(np.sum(term.levels[fixed] * term.strides[fixed]))
    offsets = np.array([base], dtype=np.int64)
    partners = offsets.copy()
    weights = np.ones(1)
    free = ~fixed
    for stride, dim, kind in zip(term.strides[free], term.dims[free], term.kinds[free]):
        digit = np.arange(dim)
        partner_digit = 1 - digit if kind in (KIND_X, KIND_Y) else digit
        if kind in (KIND_Y, KIND_Z):
            weight = np.where(digit == 1, -1.0, 1.0)
        elif kind == KIND_NUMBER:
            weight = digit.astype(float)
        else:
            weight = np.ones(dim)
        offsets = (offsets[:, None] + digit * stride).ravel()
        partners = (partners[:, None] + partner_digit * stride).ravel()
        weights = (weights[:, None] * weight).ravel()
    keep = weights != 0.0
    return offsets[keep], partners[keep], weights[keep]


def _view(
    flat: np.ndarray, start: int, shape: tuple[int, ...], steps: tuple[int, ...]
) -> np.ndarray:
    """Return a read-only view of ``flat`` from ``start`` with byte ``steps``."""
    return np.lib.stride_tricks.as_strided(
        flat[start:], shape=shape, strides=steps, writeable=False
    )


def table_term(data: np.ndarray, term: _Layout, *, density: bool) -> complex:
    """Return one term's value from its table and strided views of the state.

    Every table row fixes the named digits, which leaves the term reading one
    regular lattice of the state: the subsystems it does not name, at
    ``rest_strides``. That lattice is a view, so a row costs one dot product
    (statevector) or one sum along a shifted diagonal (density matrix), and
    nothing the size of the state is indexed. ``data`` must be C-contiguous.
    """
    # Largest stride first, so that the innermost axis is the densest.
    rest_strides = term.rest_strides[::-1]
    shape = tuple(int(dim) for dim in term.rest_dims[::-1])
    offsets, partners, weights = _table(term)
    total = 0.0 + 0.0j
    if density:
        # rho[j, p(j)] sits at j * size + p(j) in the flat matrix, and moving
        # along an unnamed subsystem moves j and p(j) together.
        size = data.shape[0]
        flat = data.reshape(-1)
        steps = tuple(int(s) * (size + 1) * data.itemsize for s in rest_strides)
        for offset, partner, weight in zip(offsets, partners, weights):
            view = _view(flat, offset * size + partner, shape, steps)
            total += weight * view.sum()
        return complex(total)
    steps = tuple(int(s) * data.itemsize for s in rest_strides)
    contiguous = len(shape) == 0 or (len(shape) == 1 and term.rest_strides[0] == 1)
    for offset, partner, weight in zip(offsets, partners, weights):
        bra = _view(data, partner, shape, steps)
        ket = _view(data, offset, shape, steps)
        if contiguous:
            total += weight * np.vdot(bra, ket)
        else:
            # vdot would copy both strided views; this allocates one product.
            total += weight * (np.conj(bra) * ket).sum()
    return complex(total)


# A term's table is read a row at a time, at a fixed Python cost per row of
# about ten microseconds, while the per-amplitude kernels cost tens of
# nanoseconds per amplitude. With at least this many amplitudes per row, the
# per-row cost stays a fraction of one per-amplitude pass, so the table wins.
AMPLITUDES_PER_ROW = 1024


def _y_count(factors: _Factors) -> int:
    """Return how many of a term's factors are ``Y``; each contributes ``i``."""
    return sum(
        isinstance(factor, _Pauli) and factor.letter == "Y" for _axis, factor in factors
    )


def _sum_terms(
    terms: tuple[tuple[float, _Factors], ...],
    term_value: Callable[[_Factors], complex],
) -> float:
    """Sum one observable's terms, each evaluated by ``term_value``.

    ``term_value`` evaluates one term's factors against an already-bound
    state, without the ``i**n_y`` phase of its Y factors, which is applied
    here; each runtime supplies its own.
    """
    total = 0.0 + 0.0j
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue  # a zero coefficient contributes nothing; skip the pass
        total += coefficient * term_value(factors) * (1j ** _y_count(factors))
    return float(total.real)


def _expectation(
    data: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
    *,
    density: bool,
) -> float:
    size = data.shape[0]
    qubits = all(dim == 2 for dim in dims)
    qubit_kernel = qubit_density_matrix_term if density else qubit_statevector_term
    tables = data.flags.c_contiguous

    @functools.cache
    def index() -> np.ndarray:
        # Only the per-amplitude kernels need it; built once, when first used.
        return np.arange(size)

    def term_value(factors: _Factors) -> complex:
        term = _layout(factors, dims, strides)
        if tables and term.rows * AMPLITUDES_PER_ROW <= size:
            return table_term(data, term, density=density)
        if qubits:
            return qubit_kernel(data, index(), *_term_masks(factors, strides))
        return digit_term(
            data,
            index(),
            term.strides,
            term.dims,
            term.kinds,
            term.levels,
            density=density,
        )

    return _sum_terms(terms, term_value)


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
