"""Structural interface for custom execution backends."""

from typing import Protocol

from .job import Job
from .program import Program
from .result import Result


class Backend(Protocol):
    """Run a program using a backend's own execution options.

    Implementations need not inherit from this protocol. Each backend defines
    and documents its supported options and the result fields it produces.
    """

    def run(self, program: Program, /) -> Job[Result]:
        """Submit one program and return its job.

        Args:
            program: Program to execute.

        Returns:
            A job containing the result produced by the backend.
        """
