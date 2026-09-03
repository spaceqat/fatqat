"""Backend-neutral observable planning and sampled statistics."""

import math

import pytest

from fatqat._expectation import (
    _TermOccurrence,
    _combine_term_statistics,
    _plan_term_occurrences,
    _reconstruct_term_outcome,
    _reduce_outcome_counts,
    _sample_mean_and_standard_error,
)
from fatqat.observable import Observable


def test_term_plan_preserves_executable_occurrence_order():
    observables = (
        Observable([("Z", 0.0), ("I", 2.0), ("X", 1.0), ("X", -0.5)]),
        Observable([("I", -1.0), ("Z", 3.0)]),
    )

    occurrences, constants = _plan_term_occurrences(observables)

    assert constants == (2.0, -1.0)
    assert occurrences == (
        _TermOccurrence(0, 1.0, ((0, "X"),)),
        _TermOccurrence(0, -0.5, ((0, "X"),)),
        _TermOccurrence(1, 3.0, ((0, "Z"),)),
    )


def test_mixed_outcome_statistics_use_unbiased_standard_error():
    factors = ((0, "Z"), (1, "ONE"))
    outcome_sum, outcome_sum_squared = _reduce_outcome_counts(
        factors,
        [
            ((0, 1), 1),  # +1
            ((1, 1), 1),  # -1
            ((0, 0), 1),  # 0
        ],
    )

    mean, standard_error = _sample_mean_and_standard_error(
        outcome_sum, outcome_sum_squared, shots=3
    )

    assert (outcome_sum, outcome_sum_squared) == (0, 2)
    assert mean == 0.0
    assert standard_error**2 * 3 == pytest.approx(1.0)
    assert standard_error == pytest.approx(math.sqrt(1 / 3))


def test_term_outcome_covers_paulis_and_projectors():
    factors = ((0, "X"), (1, "Y"), (2, "ZERO"))

    assert _reconstruct_term_outcome(factors, (0, 1, 0)) == -1
    assert _reconstruct_term_outcome(factors, (0, 1, 1)) == 0


def test_independent_term_standard_errors_combine_in_quadrature():
    observable = Observable.from_sparse(
        [("Z", (0,), 2.0), ("ONE", (0,), -3.0)],
        num_qubits=1,
    )
    occurrences, constants = _plan_term_occurrences((observable,))

    values, standard_errors = _combine_term_statistics(
        constants,
        occurrences,
        (
            (0, 2),  # outcomes +1, -1: mean 0, standard error 1
            (1, 1),  # outcomes 0, +1: mean 0.5, standard error 0.5
        ),
        shots=2,
    )

    assert values == pytest.approx((-1.5,))
    assert standard_errors == pytest.approx((2.5,))
