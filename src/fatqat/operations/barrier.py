"""Barrier: compiler-facing no-op frontend operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class BarrierGate(Operation):
    """Barrier marker: no simulation semantics, preserved for compiler passes.

    Has no matrix and no effect on any simulated state or on counts; the
    matrix-family backend recognizes it by operation type during lowering and
    skips it entirely. The frontend keeps it verbatim in
    ``Program.operations``, so compiler passes can read barrier boundaries
    from the un-lowered program.

    The class itself is not part of the ``fq.ops`` public surface (not in
    ``__all__``) but stays attribute-accessible for ``isinstance`` checks;
    ``Barrier`` (the singleton) is the one users construct programs with. A
    barrier may span any number of subsystems.

    Examples:
        A barrier between preparation and measurement changes nothing:

        >>> import fatqat as fq
        >>> program = fq.Program(2, 2)
        >>> program.add(fq.ops.X, 0)
        >>> program.add(fq.ops.Barrier, (0, 1))
        >>> program.add_measurement((0, 1), (0, 1))
        >>> result = fq.backends.SimulatorBackend().run(
        ...     program, shots=5, simulation_config={"seed": 0}
        ... ).result()
        >>> result.get_counts()
        {'01': 5}
    """

    name: ClassVar[str] = "Barrier"
    _num_subsystems: ClassVar[int | None] = None


# `Barrier` takes no parameters, so - like the fixed gates and `Reset` - it is
# exported only as a singleton value: `fq.ops.Barrier`, not `fq.ops.Barrier()`.
Barrier = BarrierGate()
