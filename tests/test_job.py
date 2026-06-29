import pytest

from qnsim.job import Job


def test_done_job_returns_result():
    job = Job.done("RESULT")
    assert job.status == "DONE"
    assert job.result() == "RESULT"


def test_error_job_reraises():
    job = Job.failed(ValueError("boom"))
    assert job.status == "ERROR"
    with pytest.raises(ValueError, match="boom"):
        job.result()
