"""Synchronous Qiskit ``JobV1`` adapter for fatqat."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.providers.job import JobStatus, JobV1
from qiskit.exceptions import QiskitError

if TYPE_CHECKING:
    from qiskit.providers.backend import BackendV2
    from qiskit.result import Result


class FatqatJob(JobV1):
    """Terminal synchronous job carrying an eager Qiskit ``Result``."""

    def __init__(
        self,
        backend: BackendV2,
        job_id: str,
        *,
        result: Result | None = None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(backend, job_id=job_id)
        self._stored_result = result
        self._error = error

    def submit(self) -> None:
        """No-op: the job is already complete when returned."""

    def status(self) -> JobStatus:
        if self._error is not None:
            return JobStatus.ERROR
        return JobStatus.DONE

    def result(self, timeout: float | None = None) -> Result:
        del timeout
        if self._error is not None:
            raise QiskitError(str(self._error)) from self._error
        if self._stored_result is None:
            raise QiskitError("job completed without a result payload")
        return self._stored_result

    def cancel(self) -> bool:
        return False

    def backend(self) -> BackendV2:
        return self._backend
