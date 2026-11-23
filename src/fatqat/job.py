"""Completed job handles returned by FATQAT execution APIs."""

from __future__ import annotations

from typing import Generic, TypeVar, cast

T = TypeVar("T")


class Job(Generic[T]):
    """Hold the result of a completed submission.

    Built-in backends and estimators return completed jobs.
    Call ``result()`` to obtain a ``Result`` for one run or a list of results
    for a sweep. If the job holds an execution error, ``result()`` raises it.

    Attributes:
        status: ``"DONE"`` for a successful submission or ``"ERROR"`` for a
            failed one. Applications should treat this value as read-only.
    """

    status: str

    def __init__(
        self,
        status: str,
        result: T | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Create a completed job.

        Most users receive jobs from a backend or estimator ``run`` method.

        Args:
            status: ``"DONE"`` for success or ``"ERROR"`` for failure.
            result: Value returned by ``result()`` for a successful job.
            error: Exception raised by ``result()`` for a failed job.
        """
        self.status = status
        self._result = result
        self._error = error

    def result(self) -> T:
        """Return the result value or raise the execution error.

        Returns:
            The value produced by a ``"DONE"`` job.

        Raises:
            BaseException: The error from an ``"ERROR"`` job.
            RuntimeError: If the job is incomplete or failed without a recorded
                error.
        """
        if self.status == "DONE":
            return cast(T, self._result)
        if self.status == "ERROR":
            if self._error is None:
                # `raise None` would surface as an unrelated TypeError.
                raise RuntimeError("job failed with no recorded error")
            raise self._error
        raise RuntimeError(f"job is not complete (status={self.status!r})")
