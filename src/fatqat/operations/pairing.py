"""Pair and Unpair connectivity instructions for ``AtomArraySimulator``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PairGate(Operation):
    """Add one undirected edge to the atom-array connectivity graph.

    ``Pair`` is implemented by ``AtomArraySimulator`` only. Targets are the
    two distinct atoms ``(a, b)``; their order does not affect connectivity.
    Pairing an already connected pair is a connectivity no-op. The instruction
    changes no quantum state and does not make an otherwise unsupported gate
    available: it only satisfies the connectivity prerequisite for a supported
    two-atom gate such as the backend's native CZ.

    ``Pair`` must be unconditional. ``Program.add`` accepts a condition
    syntactically, but ``AtomArraySimulator`` raises ``BackendValidationError``
    while lowering it. Noise attached to ``Pair`` is emitted on its targets and
    can model movement loss or decoherence even though ``Pair`` itself has no
    execution matrix.

    Add the singleton ``ops.Pair`` without parentheses. It requires exactly two
    distinct scalar targets and rejects ``RegisterView``. This implementation
    class is not exported through ``fatqat.operations.__all__``.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Pair, (0, 1))
    """

    name: ClassVar[str] = "Pair"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class UnpairGate(Operation):
    """Remove one undirected edge from the atom-array connectivity graph.

    ``Unpair`` is implemented by ``AtomArraySimulator`` only. Targets are the
    two distinct atoms ``(a, b)``; their order does not affect connectivity.
    Removing an absent edge is a connectivity no-op. The instruction changes
    no quantum state, but attached movement-cost noise is still emitted on its
    targets.

    ``Unpair`` must be unconditional. ``Program.add`` accepts a condition
    syntactically, but ``AtomArraySimulator`` raises ``BackendValidationError``
    while lowering it. After unpairing, a supported two-atom gate on that pair
    again fails its connectivity check until another ``Pair``.

    Add the singleton ``ops.Unpair`` without parentheses. It requires exactly
    two distinct scalar targets and rejects ``RegisterView``. This
    implementation class is not exported through
    ``fatqat.operations.__all__``.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Unpair, (0, 1))
    """

    name: ClassVar[str] = "Unpair"
    num_subsystems: ClassVar[int] = 2


# singleton values: `ops.Pair` / `ops.Unpair`, not `ops.Pair()`.
Pair = PairGate()
Unpair = UnpairGate()
