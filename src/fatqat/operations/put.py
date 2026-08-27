"""Put: add atoms into target sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PutGate(Operation):
    """Add a fresh ``|0>`` atom into each target site that is currently empty.

    ``Put`` is the atom-introduction instruction. Per shot, each target holding
    no atom gets a fresh atom in ``|0>``, fully usable afterwards (gates apply;
    measurement reads a normal digit, not the erasure marker); a target that
    already holds an atom is left untouched. Every site starts empty and is
    populated by ``Put``, which may appear any number of times and target any
    sites.

    It has no matrix; the neutral-atom backend resolves it by type to a per-shot
    fill step, like ``Reset``. Imperfect loading efficiency is expressed by
    attaching ``Loss`` to ``Put`` (the atom arrives, then may be lost), not
    by a success-rate parameter. Occupancy/count is owned entirely by the
    engine's per-shot occupancy state; ``Put`` only marks targets present from
    this point.

    Variable arity (one or more targets). Exposed as the singleton ``ops.Put``;
    the class stays attribute-accessible for ``isinstance`` checks.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(3)
        >>> program.add(ops.Put, (0, 1, 2))
        >>> program.operations[0].operation.name
        'Put'
        >>> program.operations[0].operation.num_subsystems is None
        True
    """

    name: ClassVar[str] = "Put"
    num_subsystems: ClassVar[int | None] = None


# singleton value: `ops.Put`, not `ops.Put()`.
Put = PutGate()
