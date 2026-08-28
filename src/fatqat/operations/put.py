"""Put: add atoms into target sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PutGate(Operation):
    """Load a fresh ``|0>`` atom into each empty target site.

    ``Put`` is implemented only by `fatqat.simulator.AtomArraySimulator`. If a
    program uses it, every site starts empty on each shot and only ``Put``
    loads it. An occupied target is unchanged, and a later ``Put`` can reload a
    lost atom in ``|0>``. Other built-in backends raise
    `fatqat.errors.UnsupportedOperationError`.

    Model loading efficiency by attaching `fatqat.noise.Loss` to `Put`.
    `fatqat.NoiseModel.add` raises `ValueError` for any other noise declaration.
    Loss is evaluated after every matching `Put` operation whose condition
    passes, even when its target was already occupied. It shares the `Put`
    condition. `Put` itself has no success-rate argument.

    ``Put`` accepts one or more distinct scalar targets and can carry a
    condition. `fatqat.Program.add` rejects
    `fatqat.RegisterView` and an empty target tuple.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(3)
        >>> program.add(ops.Put, (0, 1, 2))
    """

    name: ClassVar[str] = "Put"
    num_subsystems: ClassVar[int | None] = None


Put = PutGate()
