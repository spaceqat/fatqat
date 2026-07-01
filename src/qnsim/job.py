"""Eager Job handle: DONE carries a Result, ERROR re-raises on result()."""

from __future__ import annotations


class Job:
    """Eager job handle returned by backends.

    Phase 1 jobs are already terminal when returned. `DONE` jobs return their
    result, while `ERROR` jobs re-raise their stored exception from `result()`.
    """

    def __init__(self, status: str, result=None, error: BaseException | None = None):
        """Create a job with a terminal or non-terminal status.

        Args:
            status: Job status string.
            result: Result payload for `DONE` jobs.
            error: Exception payload for `ERROR` jobs.
        """
        self.status = status
        self._result = result
        self._error = error

    @classmethod
    def done(cls, result) -> "Job":
        """Create a completed job carrying `result`."""
        return cls(status="DONE", result=result)

    @classmethod
    def failed(cls, error: BaseException) -> "Job":
        """Create an error job carrying `error`."""
        return cls(status="ERROR", error=error)

    def result(self):
        """Return the result payload or raise the terminal job error.

        Raises:
            BaseException: Re-raises the stored error for `ERROR` jobs.
            RuntimeError: If the job is not in a terminal state.
        """
        if self.status == "DONE":
            return self._result
        if self.status == "ERROR":
            raise self._error
        raise RuntimeError(f"job is not complete (status={self.status!r})")
