"""Reset: non-unitary frontend operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reset operation: repreparation of one or more target subsystems in ``|0>``.

    Has no matrix; the matrix-family backend resolves it to a boundary reset
    step by operation type. The class itself is not part of the ``qs.ops``
    public surface (not in ``__all__``) but stays attribute-accessible for
    ``isinstance`` checks against ``Reset`` steps; ``Reset`` (the singleton)
    is the one users construct programs with.
    """

    name: ClassVar[str] = "Reset"
    _num_subsystems: ClassVar[int | None] = None


# `Reset` takes no parameters, so - like the fixed gates - it is exported only
# as a singleton value: `qs.ops.Reset`, not `qs.ops.Reset()`.
Reset = ResetGate()
