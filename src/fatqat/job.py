"""Generic eager Job handle with terminal success and error states."""

from __future__ import annotations

from typing import Generic, TypeVar, cast

T = TypeVar("T")


class Job(Generic[T]):
    """Eager job handle returned by backends.

    Phase 1 jobs are already terminal when returned. ``DONE`` jobs return
    their result, while ``ERROR`` jobs re-raise their stored exception from
    ``result()``.

    Attributes:
        status: Current eager terminal status, normally ``"DONE"`` or
            ``"ERROR"``.
    """

    status: str

    def __init__(
        self,
        status: str,
        result: T | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Create a job with a terminal or non-terminal status.

        Args:
            status: Job status string.
            result: Payload for `DONE` jobs.
            error: Exception payload for `ERROR` jobs.
        """
        self.status = status
        self._result = result
        self._error = error

    def result(self) -> T:
        """Return the result payload or raise the terminal job error.

        Returns:
            The result payload stored on a ``DONE`` job.

        Raises:
            BaseException: Re-raises the stored error for ``ERROR`` jobs.
            RuntimeError: If the job is not in a terminal state.
        """
        if self.status == "DONE":
            return cast(T, self._result)
        if self.status == "ERROR":
            raise self._error
        raise RuntimeError(f"job is not complete (status={self.status!r})")
