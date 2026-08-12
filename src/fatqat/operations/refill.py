"""Refill: atom-grid reload of empty sites with fresh |0> atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class RefillGate(Operation):
    """Reload atoms into the target sites that are currently empty.

    Per shot, each target site holding no atom gets a fresh atom in ``|0>``,
    fully usable afterward (gates apply, measurement reads a normal digit, not
    the erasure marker); a target that already holds an atom is left untouched
    -- a full trap cannot take another atom (M-C2). Refill may fill a site
    ``LoadAtoms`` never loaded, not only a loss-cleared one -- the reservoir
    does not care why the site is empty (M-C4).

    Has no matrix; the atom-grid backend resolves it to a per-shot refill step
    by operation type, exactly like ``Reset``. Imperfect loading efficiency is
    expressed by attaching ``AtomLoss`` to ``Refill`` (the atom arrives, then
    may be lost), not by a success-rate parameter (S-C1). The class is not part
    of the public ``fatqat.operations`` surface but stays attribute-accessible
    for isinstance checks; ``Refill`` (the singleton) is what users build with.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as op
        >>> atoms = fq.GridRegister(1, 2, name="atoms")
        >>> program = fq.Program([atoms])
        >>> program.add(op.LoadAtoms(1, 2))
        >>> program.add(op.Refill, (atoms[0], atoms[1]))
        >>> program.operations[1].operation.name
        'Refill'
    """

    name: ClassVar[str] = "Refill"
    _num_subsystems: ClassVar[int | None] = None


# Like Reset, Refill takes no parameters and is exported as a singleton value.
Refill = RefillGate()