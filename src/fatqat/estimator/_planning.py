"""Backend-neutral observable planning, validation, and sampling statistics.

A sampled expectation splits each observable into ordered term occurrences,
reconstructs one term outcome per shot from measured digits, and combines the
independent term estimates into a value and a standard error. Exact evaluation
lives in :mod:`._exact`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..errors import BackendValidationError
from ..observable import NumberOperator, Observable, Projector, _Factors, _Pauli

if TYPE_CHECKING:
    from ..program import Program


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
            f"got {len(measured_digits)} measured digits for {len(factors)} operators"
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
            raise ValueError(f"unsupported measured operator {factor!r}")
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
    observables: Sequence[Observable],
    program: Program,
    subsystem_dims: Sequence[int | None] | None = None,
) -> None:
    """Check every local operator against the program subsystem it names.

    Observables carry no width or dimension, so this is where an index past
    the program, a Pauli on a non-qubit, or a projector level the subsystem
    cannot hold is rejected. A subsystem with no fixed dimension (a mode) is
    never a qubit; its projector levels are checked once a backend passes the
    ``subsystem_dims`` it resolved, one per program subsystem.
    """
    if subsystem_dims is None:
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
                    has = "no fixed dimension" if dim is None else f"dim={dim}"
                    raise BackendValidationError(
                        f"local operator {factor} on subsystem {index} is "
                        f"defined on qubits only; the subsystem has {has}"
                    )
                if (
                    isinstance(factor, Projector)
                    and dim is not None
                    and factor.level >= dim
                ):
                    raise BackendValidationError(
                        f"local operator {factor!r} on subsystem {index} needs "
                        f"dim > {factor.level}, but the subsystem has dim={dim}"
                    )
