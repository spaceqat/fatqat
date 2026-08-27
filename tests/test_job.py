"""Tests eager Job terminal-state behavior."""

import pytest

from fatqat.job import Job


def test_done_job_returns_result():
    job = Job(status="DONE", result="RESULT")
    assert job.status == "DONE"
    assert job.result() == "RESULT"


def test_done_job_returns_same_ordered_list_payload():
    results = [object(), object()]

    job = Job(status="DONE", result=results)

    assert job.result() is results
    assert job.result() == [results[0], results[1]]


def test_error_job_reraises():
    job = Job(status="ERROR", error=ValueError("boom"))
    assert job.status == "ERROR"
    with pytest.raises(ValueError, match="boom"):
        job.result()


def test_error_job_reraises_base_exception_payload():
    error = KeyboardInterrupt("stopped")
    job = Job(status="ERROR", error=error)

    with pytest.raises(KeyboardInterrupt, match="stopped") as caught:
        job.result()
    assert caught.value is error


@pytest.mark.parametrize("name", ["done", "failed", "from_result", "from_error"])
def test_job_convenience_factories_are_not_exposed(name):
    assert not hasattr(Job, name)


def test_result_raises_for_non_terminal_status():
    job = Job(status="PENDING")
    with pytest.raises(RuntimeError, match="not complete"):
        job.result()


def test_no_sweep_job_type_is_exposed():
    import fatqat as fq

    assert not hasattr(fq, "SweepJob")
