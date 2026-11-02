"""Backend-neutral exact expectation evaluation.

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
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence

import numpy as np

from ..observable import NumberOperator, Projector, _Factors, _Pauli
from ._numpy_kernels import (
    KIND_NUMBER,
    KIND_PROJECTOR,
    KIND_X,
    KIND_Y,
    KIND_Z,
    digit_term,
    qubit_density_matrix_term,
    qubit_statevector_term,
)

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
    from . import _numba_kernels

    return (
        _numba_kernels.qubit_statevector_term,
        _numba_kernels.qubit_density_matrix_term,
        _numba_kernels.statevector_term,
        _numba_kernels.density_matrix_term,
    )


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


def _bind_digit_evaluator(
    data: np.ndarray, *, density: bool
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], complex]:
    """Return ``arrays -> value`` for one mixed-dimension state."""
    if _COMPILED is not None:
        kernel = _COMPILED[3 if density else 2]
        return lambda *arrays: kernel(data, *arrays)
    index = np.arange(data.shape[0])
    return lambda *arrays: digit_term(data, index, *arrays, density=density)


def _expectation(
    data: np.ndarray,
    terms: tuple[tuple[float, _Factors], ...],
    dims: Sequence[int],
    strides: Sequence[int],
    *,
    density: bool,
) -> float:
    """Sum one observable's terms on a statevector or density matrix."""
    total = 0.0 + 0.0j
    if all(dim == 2 for dim in dims):
        term_value = _bind_term_evaluator(
            data,
            1 if density else 0,
            qubit_density_matrix_term if density else qubit_statevector_term,
        )
        for coefficient, factors in terms:
            if coefficient == 0.0:
                continue  # a zero coefficient contributes nothing; skip the pass
            x_mask, z_mask, zero_mask, one_mask, n_y = _term_masks(factors, strides)
            value = term_value(x_mask, z_mask, zero_mask, one_mask)
            total += coefficient * value * (1j**n_y)
        return float(total.real)

    digit_value = _bind_digit_evaluator(data, density=density)
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue
        *arrays, n_y = _term_digits(factors, dims, strides)
        total += coefficient * digit_value(*arrays) * (1j**n_y)
    return float(total.real)


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
