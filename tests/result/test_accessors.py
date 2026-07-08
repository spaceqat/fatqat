"""Tests result configuration, count-key assembly, and result accessors."""

import numpy as np
import pytest

from fatqat.result import Result, decode_indices_to_clbit_rows
from fatqat.errors import ResultFieldUnavailableError


def test_decode_indices_little_endian_key():
    rows = decode_indices_to_clbit_rows(
        [1, 1, 0], measurements=[(0, 0), (1, 1)], system_dims=(2, 2), n_clbits=2
    )
    assert np.array_equal(rows, np.array([[1, 0], [1, 0], [0, 0]], dtype=int))


def test_decode_indices_unwritten_clbit_stays_zero():
    rows = decode_indices_to_clbit_rows(
        [1, 1], measurements=[(0, 0)], system_dims=(2,), n_clbits=2
    )
    assert np.array_equal(rows, np.array([[1, 0], [1, 0]], dtype=int))


def test_decode_indices_last_write_wins():
    rows = decode_indices_to_clbit_rows(
        [2], measurements=[(0, 0), (1, 0)], system_dims=(2, 2), n_clbits=2
    )
    assert np.array_equal(rows, np.array([[1, 0]], dtype=int))


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
    assert r.get_counts() == {"10": 5, "02": 3}


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
