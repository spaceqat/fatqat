"""Tests result configuration, count-key assembly, and result accessors."""

import numpy as np
import pytest

from fatqcat.result import Result, build_counts
from fatqcat.errors import ResultFieldUnavailableError


def test_build_counts_little_endian_key():
    # 2 clbits; measurement maps qubit0->clbit0, qubit1->clbit1
    # index 1 = qubit0=1, qubit1=0 -> clbit0=1,clbit1=0 -> key (1, 0)
    indices = [1, 1, 0]
    counts = build_counts(
        indices, n_clbits=2, measurements=[(0, 0), (1, 1)], system_dims=(2, 2)
    )
    assert counts == {(1, 0): 2, (0, 0): 1}


def test_build_counts_unwritten_clbit_stays_zero():
    # 2 clbits but only clbit0 written from qubit0
    indices = [1, 1]
    counts = build_counts(
        indices, n_clbits=2, measurements=[(0, 0)], system_dims=(2,)
    )
    assert counts == {(1, 0): 2}


def test_build_counts_last_write_wins():
    # both measurements target clbit0; second uses qubit1
    # index 2 = qubit1=1, qubit0=0 -> clbit0 set by qubit1 -> 1 -> key (1, 0)
    counts = build_counts(
        [2], n_clbits=2, measurements=[(0, 0), (1, 0)], system_dims=(2, 2)
    )
    assert counts == {(1, 0): 1}


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
