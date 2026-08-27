"""Non-unitary reset operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reprepare one or more target subsystems in ``|0>``.

    Reset is a non-unitary instruction with no matrix or attachable noise
    boundary. It can be conditioned through ``Program.add``. Statevector
    execution samples the reset branch when the target is entangled; density
    matrix execution applies the corresponding deterministic channel.

    Add the singleton ``ops.Reset`` without parentheses. It accepts one or
    more distinct scalar targets of any local dimension; ``RegisterView`` and
    an empty target tuple are rejected by ``Program.add``.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(1)
        >>> program.add(ops.X, 0)
        >>> program.add(ops.Reset, 0)
        >>> result = fq.simulator.Simulator("SV").run(
        ...     program, shots=1,
        ...     result_config={"counts": False, "final_state": True},
        ... ).result()
        >>> result.get_statevector()
        array([1.+0.j, 0.+0.j])
    """

    name: ClassVar[str] = "Reset"
    num_subsystems: ClassVar[int | None] = None


# `Reset` takes no parameters, so - like the fixed gates - it is exported only
# as a singleton value: `ops.Reset`, not `ops.Reset()`.
Reset = ResetGate()
