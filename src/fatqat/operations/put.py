"""Put: add atoms into target sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PutGate(Operation):
    """Load a fresh ``|0>`` atom into each empty target site.

    ``Put`` is implemented by ``AtomArraySimulator`` only. If a program uses
    any ``Put``, every declared site starts empty for each shot and must be
    populated explicitly. A target that is already occupied is left in its
    current quantum state. A later ``Put`` can reload a lost atom in ``|0>``.
    Other built-in matrix and pulse backends report the operation as
    unsupported.

    Loading efficiency is modeled by attaching ``fatqat.noise.Loss`` to
    ``Put``; no other noise declaration may use this boundary. The loss is
    evaluated after every matching ``Put`` occurrence whose condition passes,
    including one whose target was already occupied. It shares the ``Put``
    condition. ``Put`` itself has no success-rate argument.

    Add the singleton ``ops.Put`` without parentheses. It accepts one or more
    distinct scalar targets and supports ``Program.add(condition=...)``;
    ``RegisterView`` and an empty target tuple are rejected.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(3)
        >>> program.add(ops.Put, (0, 1, 2))
    """

    name: ClassVar[str] = "Put"
    num_subsystems: ClassVar[int | None] = None


# singleton value: `ops.Put`, not `ops.Put()`.
Put = PutGate()
