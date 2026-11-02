"""Backend-neutral observable planning, statistics, and exact kernels.

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
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .errors import BackendValidationError
from .observable import NumberOperator, Observable, Projector, _Factors, _Pauli

if TYPE_CHECKING:
    from .program import Program

# (letter -> contributes to) masks. Y is both a bit flip and a sign, which is
# exactly the X*Z decomposition above.
_FLIPS = frozenset({"X", "Y"})
_SIGNS = frozenset({"Y", "Z"})


@dataclass(frozen=True, slots=True)
class _TermOccurrence:
    """One executable term in its stable public order."""

    observable_index: int
    coefficient: float
    logical_factors: _Factors


@dataclass(frozen=True, slots=True)
class _ExpectationExecution:
    """Ordered backend values and uncertainty facts for one request."""

    values: tuple[float, ...]
    standard_errors: tuple[float, ...]
    method: str
    runtime: str
    solver: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "standard_errors", tuple(self.standard_errors))


def _plan_term_occurrences(
    observables: Sequence[Observable],
) -> tuple[tuple[_TermOccurrence, ...], tuple[float, ...]]:
    """Separate exact constants from executable stored term occurrences."""
    constants = [0.0] * len(observables)
    occurrences = []
    for observable_index, observable in enumerate(observables):
        for coefficient, factors in observable.terms:
            if coefficient == 0.0:
                continue
            if not factors:
                constants[observable_index] += coefficient
                continue
            occurrences.append(_TermOccurrence(observable_index, coefficient, factors))
    return tuple(occurrences), tuple(constants)


def _reconstruct_term_outcome(
    factors: _Factors,
    measured_digits: Sequence[int],
) -> int:
    """Return the product outcome for factor-aligned level measurements.

    A Pauli factor is measured in its eigenbasis, so its digit is binary; a
    projector or number factor reads the measured level directly.
    """
    if len(factors) != len(measured_digits):
        raise ValueError(
            f"got {len(measured_digits)} measured digits for {len(factors)} factors"
        )

    outcome = 1
    for (_, factor), raw_digit in zip(factors, measured_digits, strict=True):
        if (
            isinstance(raw_digit, (bool, np.bool_))
            or not isinstance(raw_digit, (int, np.integer))
            or raw_digit < 0
        ):
            raise ValueError(
                f"measurement digit must be a non-negative integer, got {raw_digit!r}"
            )
        digit = int(raw_digit)
        if isinstance(factor, _Pauli):
            if digit not in (0, 1):
                raise ValueError(f"measurement digit must be binary, got {digit!r}")
            outcome *= 1 if digit == 0 else -1
        elif isinstance(factor, Projector):
            outcome *= 1 if digit == factor.level else 0
        elif isinstance(factor, NumberOperator):
            outcome *= digit
        else:
            raise ValueError(f"unsupported measured factor {factor!r}")
    return outcome


def _reduce_outcome_counts(
    factors: _Factors,
    outcome_counts: Iterable[tuple[Sequence[int], int]],
) -> tuple[int, int]:
    """Reduce typed outcome counts to their first and second raw sums."""
    outcome_sum = 0
    outcome_sum_squared = 0
    for measured_digits, raw_count in outcome_counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"outcome count must be nonnegative, got {raw_count!r}")
        outcome = _reconstruct_term_outcome(factors, measured_digits)
        outcome_sum += count * outcome
        outcome_sum_squared += count * outcome * outcome
    return outcome_sum, outcome_sum_squared


def _sample_mean_and_standard_error(
    outcome_sum: int,
    outcome_sum_squared: int,
    *,
    shots: int,
) -> tuple[float, float]:
    """Return a sample mean and its unbiased standard error."""
    if shots < 1:
        raise ValueError(f"sample statistics require shots >= 1, got {shots}")
    mean = outcome_sum / shots
    if shots == 1:
        return mean, math.nan

    numerator = shots * outcome_sum_squared - outcome_sum * outcome_sum
    standard_error = math.sqrt(numerator / (shots * shots * (shots - 1)))
    return mean, standard_error


def _combine_term_statistics(
    constants: Sequence[float],
    occurrences: Sequence[_TermOccurrence],
    statistics: Sequence[tuple[int, int]],
    *,
    shots: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Combine independent sampled occurrences into ordered observables."""
    if shots < 1:
        raise ValueError(f"sample statistics require shots >= 1, got {shots}")
    if len(occurrences) != len(statistics):
        raise ValueError(
            f"got {len(statistics)} statistics for {len(occurrences)} occurrences"
        )

    values = list(constants)
    variances = [0.0] * len(constants)
    has_executable = [False] * len(constants)
    for occurrence, (outcome_sum, outcome_sum_squared) in zip(
        occurrences, statistics, strict=True
    ):
        observable_index = occurrence.observable_index
        has_executable[observable_index] = True
        mean, standard_error = _sample_mean_and_standard_error(
            outcome_sum,
            outcome_sum_squared,
            shots=shots,
        )
        values[observable_index] += occurrence.coefficient * mean
        if shots > 1:
            variances[observable_index] += occurrence.coefficient**2 * standard_error**2

    standard_errors = tuple(
        math.nan if shots == 1 and executable else math.sqrt(variance)
        for executable, variance in zip(has_executable, variances, strict=True)
    )
    return tuple(values), standard_errors


def _validate_observable_targets(
    observables: Sequence[Observable], program: Program
) -> None:
    """Check every factor against the program subsystem it names.

    Observables carry no width or dimension, so this is where an index past
    the program, a Pauli on a non-qubit, or a projector level the subsystem
    cannot hold is rejected.
    """
    subsystem_dims = tuple(
        register.dim
        for register in program.quantum_registers
        for _ in range(register.size)
    )
    width = len(subsystem_dims)
    for observable in observables:
        for _, factors in observable.terms:
            for index, factor in factors:
                if index >= width:
                    raise BackendValidationError(
                        f"observable names subsystem {index}, but the program "
                        f"has {width} subsystem(s)"
                    )
                dim = subsystem_dims[index]
                if isinstance(factor, _Pauli) and dim != 2:
                    raise BackendValidationError(
                        f"observable factor {factor} on subsystem {index} is "
                        f"defined on qubits only; the subsystem has dim={dim}"
                    )
                if isinstance(factor, Projector) and factor.level >= dim:
                    raise BackendValidationError(
                        f"observable factor {factor!r} on subsystem {index} needs "
                        f"dim > {factor.level}, but the subsystem has dim={dim}"
                    )


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
        elif isinstance(factor, NumberOperator) or factor == Projector(1):
            one_mask |= bit
        elif factor == Projector(0):
            zero_mask |= bit
        else:
            raise ValueError(f"factor {factor!r} has no qubit form")
    return x_mask, z_mask, zero_mask, one_mask, n_y


# Digit-kernel factor kinds. Y flips like X and signs like Z.
_KIND_X, _KIND_Y, _KIND_Z, _KIND_PROJECTOR, _KIND_NUMBER = range(5)


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
            kind = {"X": _KIND_X, "Y": _KIND_Y, "Z": _KIND_Z}[factor.letter]
            n_y += factor.letter == "Y"
        elif isinstance(factor, Projector):
            kind = _KIND_PROJECTOR
            level = factor.level
        else:
            kind = _KIND_NUMBER
        kinds.append(kind)
        levels.append(level)
    return (
        np.array(factor_strides, dtype=np.int64),
        np.array(factor_dims, dtype=np.int64),
        np.array(kinds, dtype=np.int64),
        np.array(levels, dtype=np.int64),
        n_y,
    )


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


def _digit_term_numpy(
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
        if kind in (_KIND_X, _KIND_Y):
            partner += stride * (1 - 2 * digit)
        if kind in (_KIND_Y, _KIND_Z):
            weight = weight * np.where(digit == 1, -1.0, 1.0)
        elif kind == _KIND_PROJECTOR:
            weight = weight * (digit == level)
        elif kind == _KIND_NUMBER:
            weight = weight * digit
    if density:
        return complex(np.sum(weight * data[index, partner]))
    return complex(np.vdot(data[partner], weight * data))


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
    from . import _expectation_nb

    return (
        _expectation_nb.qubit_statevector_term,
        _expectation_nb.qubit_density_matrix_term,
        _expectation_nb.statevector_term,
        _expectation_nb.density_matrix_term,
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
    return lambda *arrays: _digit_term_numpy(data, index, *arrays, density=density)


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
            _density_matrix_term_numpy if density else _statevector_term_numpy,
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
