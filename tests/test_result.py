import numpy as np
import pytest

from qnsim.result import ResultConfig, Result, build_counts
from qnsim.errors import ResultFieldUnavailableError


def test_resultconfig_defaults():
    rc = ResultConfig()
    assert rc.counts is None
    assert rc.statevector is None


def test_build_counts_little_endian_key():
    # 2 clbits; measurement maps qubit0->clbit0, qubit1->clbit1
    # index 1 = qubit0=1, qubit1=0 -> clbit0=1,clbit1=0 -> key "01"
    indices = [1, 1, 0]
    counts = build_counts(indices, n_clbits=2, measurements=[(0, 0), (1, 1)])
    assert counts == {"01": 2, "00": 1}


def test_build_counts_unwritten_clbit_stays_zero():
    # 2 clbits but only clbit0 written from qubit0
    indices = [1, 1]
    counts = build_counts(indices, n_clbits=2, measurements=[(0, 0)])
    assert counts == {"01": 2}


def test_build_counts_last_write_wins():
    # both measurements target clbit0; second uses qubit1
    # index 2 = qubit1=1, qubit0=0 -> clbit0 set by qubit1 -> 1 -> key "01"
    counts = build_counts([2], n_clbits=2, measurements=[(0, 0), (1, 0)])
    assert counts == {"01": 1}


def test_result_get_counts_available():
    r = Result(counts={"00": 5}, available=frozenset({"counts"}))
    assert r.get_counts() == {"00": 5}


def test_result_get_counts_unavailable_raises():
    r = Result(available=frozenset())
    with pytest.raises(ResultFieldUnavailableError):
        r.get_counts()


def test_result_get_statevector_unavailable_raises():
    r = Result(available=frozenset({"counts"}))
    with pytest.raises(ResultFieldUnavailableError):
        r.get_statevector()
