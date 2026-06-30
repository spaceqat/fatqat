"""Eager Job handle: DONE carries a Result, ERROR re-raises on result()."""

from __future__ import annotations


class Job:
    def __init__(self, status: str, result=None, error: BaseException | None = None):
        self.status = status
        self._result = result
        self._error = error

    @classmethod
    def done(cls, result) -> "Job":
        return cls(status="DONE", result=result)

    @classmethod
    def failed(cls, error: BaseException) -> "Job":
        return cls(status="ERROR", error=error)

    def result(self):
        if self.status == "DONE":
            return self._result
        if self.status == "ERROR":
            raise self._error
        raise RuntimeError(f"job is not complete (status={self.status!r})")
