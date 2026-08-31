"""Tests result configuration, count-key assembly, and result accessors."""

import pytest

from fatqat.result import Result
from fatqat.errors import ResultFieldUnavailableError


def test_result_get_counts_available():
    r = Result(
        counts={(0, 0): 5}, classical_dims=(2, 2), available=frozenset({"counts"})
    )
    assert r.get_counts() == {"00": 5}


def test_result_get_counts_and_tuples():
    r = Result(
        counts={(0, 1): 5, (2, 0): 3},
        classical_dims=(3, 3),
        available=frozenset({"counts"}),
    )
    assert r.get_counts_as_tuples() == {(0, 1): 5, (2, 0): 3}
    assert r.get_counts() == {"01": 5, "20": 3}


def test_result_metadata_defaults_and_is_copied():
    metadata = {"shots": 10}
    r = Result(metadata=metadata)
    metadata["shots"] = 20
    assert r.metadata == {"shots": 10}


def test_result_get_counts_unavailable_raises():
    r = Result(available=frozenset())
    with pytest.raises(ResultFieldUnavailableError):
        r.get_counts()


def test_result_get_statevector_unavailable_raises():
    r = Result(available=frozenset({"counts"}))
    with pytest.raises(ResultFieldUnavailableError):
        r.get_statevector()
