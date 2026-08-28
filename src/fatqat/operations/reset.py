"""Non-unitary reset operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reprepare one or more target subsystems in ``|0>``.

    Reset is non-unitary and has no matrix. Selecting ``ops.Reset`` in
    `fatqat.NoiseModel.add` raises `ValueError` because reset cannot carry
    operation-scoped noise. It can be conditioned through `fatqat.Program.add`
    when the backend supports feedforward.
    For an entangled target, a statevector run samples one reset branch, while
    a density-matrix run represents the resulting mixture directly.

    Reset accepts one or more distinct scalar targets of any local dimension.
    ``Program.add`` rejects
    `fatqat.RegisterView` and an empty target tuple.

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


Reset = ResetGate()
