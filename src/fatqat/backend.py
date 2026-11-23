"""Structural interface for custom execution backends."""

from typing import Protocol

from .job import Job
from .program import Program
from .result import Result


class Backend(Protocol):
    """Protocol for executing a program and returning a job.

    A backend satisfies the protocol if it provides ``run(program)`` and
    returns a ``Job[Result]``. It can offer additional keyword options and
    defines which fields it produces. Inheriting from this class is optional.
    """

    def run(self, program: Program, /) -> Job[Result]:
        """Submit one program and return its job.

        Args:
            program: Program to execute.

        Returns:
            A job whose ``result()`` returns the data produced by the backend
            or raises the error recorded in the job.
        """
