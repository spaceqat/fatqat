"""Pair / Unpair: neutral-atom connectivity edits.

``Pair`` records a connection between two atoms and ``Unpair`` removes it, so a
two-qubit gate is legal on a connected pair. Two-qubit-gate legality follows a
dynamic graph maintained privately by the atom-array simulator.

Both are "layout-class" instructions: they carry no matrix, change only the
connectivity (never the quantum state), and emit no execution step of their own
-- the backend resolves them by type while lowering. They must be
unconditional; a conditional Pair/Unpair is rejected at lowering. Channel noise
attached to them models the physical cost of moving atoms together/apart (e.g.
``Loss`` or a decoherence channel) and is still emitted for the involved
atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PairGate(Operation):
    """Connect two atoms so a two-qubit gate becomes legal on the pair.

    Symmetric and state-preserving: ``Pair`` on ``(a, b)`` adds the single
    undirected edge ``a``-``b`` to the connectivity graph and touches no third
    atom. It has no matrix; the neutral-atom backend resolves it by type,
    updating connectivity for the following segment and emitting any attached
    movement-cost noise. It must be unconditional (rejected at lowering
    otherwise). The two targets must be distinct -- ``Program.add`` already
    rejects a repeated target.

    ``Pair`` is not a gate: it changes only which pairs may interact, never the
    state. It is exposed as the singleton ``ops.Pair`` (like ``ops.Reset``); the
    class stays attribute-accessible for ``isinstance`` checks.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Put, (0, 1))
        >>> program.add(ops.Pair, (0, 1))
        >>> program.operations[1].operation.name
        'Pair'
        >>> program.operations[1].operation.num_subsystems
        2
    """

    name: ClassVar[str] = "Pair"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class UnpairGate(Operation):
    """Disconnect two atoms, removing their connectivity edge.

    The mirror of ``Pair``: removes the single undirected edge ``a``-``b`` and
    touches no third atom. Removing an edge that is not present is a silent
    no-op at the connectivity level. Same constraints as ``Pair``: no matrix,
    unconditional, state-preserving, two distinct targets, and it may carry
    movement-cost noise. Exposed as the singleton ``ops.Unpair``.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Put, (0, 1))
        >>> program.add(ops.Pair, (0, 1))
        >>> program.add(ops.Unpair, (0, 1))
        >>> program.operations[2].operation.name
        'Unpair'
    """

    name: ClassVar[str] = "Unpair"
    num_subsystems: ClassVar[int] = 2


# singleton values: `ops.Pair` / `ops.Unpair`, not `ops.Pair()`.
Pair = PairGate()
Unpair = UnpairGate()
