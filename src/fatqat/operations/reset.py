"""Reset: non-unitary frontend operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reset operation: repreparation of one or more target subsystems in ``|0>``.

    Has no matrix; the matrix-family backend resolves it to a boundary reset
    step by operation type. The class itself is not part of the ``fq.ops``
    public surface (not in ``__all__``) but stays attribute-accessible for
    ``isinstance`` checks against ``Reset`` steps; ``Reset`` (the singleton)
    is the one users construct programs with.

    Examples:
        Flip a qubit to ``|1>`` then reset it back to ``|0>``:

        >>> import fatqat as fq
        >>> program = fq.Program(1)
        >>> program.add(fq.ops.X, 0)
        >>> program.add(fq.ops.Reset, 0)
        >>> result = fq.backends.StateVectorBackend().run(
        ...     program,
        ...     shots=1,
        ...     result_config={"counts": False, "statevector": True},
        ... ).result()
        >>> result.get_statevector()
        array([1.+0.j, 0.+0.j])
    """

    name: ClassVar[str] = "Reset"
    _num_subsystems: ClassVar[int | None] = None


# `Reset` takes no parameters, so - like the fixed gates - it is exported only
# as a singleton value: `fq.ops.Reset`, not `fq.ops.Reset()`.
Reset = ResetGate()
