"""Terminal Qiskit job returned by the FATQAT adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.providers.job import JobStatus, JobV1
from qiskit.exceptions import QiskitError

if TYPE_CHECKING:
    from qiskit.providers.backend import BackendV2
    from qiskit.result import Result


class FatqatJob(JobV1):
    """Completed Qiskit job returned by :class:`FatqatBackend`.

    The job is already ``DONE`` or ``ERROR`` when returned; it never enters a
    queued or running state.
    """

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
        """Do nothing because execution finished before the job was returned."""

    def status(self) -> JobStatus:
        """Return Qiskit's ``DONE`` or ``ERROR`` terminal status."""
        if self._error is not None:
            return JobStatus.ERROR
        return JobStatus.DONE

    def result(self, timeout: float | None = None) -> Result:
        """Return the Qiskit result or raise ``QiskitError``.

        Args:
            timeout: Accepted for Qiskit API compatibility and ignored because
                the job is already terminal.

        Raises:
            qiskit.exceptions.QiskitError: If conversion or execution failed,
                or the job has no result payload. The original failure is
                chained when available.
        """
        del timeout
        if self._error is not None:
            raise QiskitError(str(self._error)) from self._error
        if self._stored_result is None:
            raise QiskitError("job completed without a result payload")
        return self._stored_result

    def cancel(self) -> bool:
        """Return ``False`` because a terminal job cannot be cancelled."""
        return False

    def backend(self) -> BackendV2:
        """Return the backend that created this job."""
        return self._backend
