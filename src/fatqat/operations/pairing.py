"""Pair and Unpair connectivity instructions for
`fatqat.simulator.AtomArraySimulator`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class PairGate(Operation):
    """Add one undirected edge to the atom-array connectivity graph.

    ``Pair`` is implemented only by `fatqat.simulator.AtomArraySimulator`;
    other built-in backends raise `fatqat.errors.UnsupportedOperationError`.
    Targets are the two distinct atoms ``(a, b)``; their order does not affect
    connectivity.
    Pairing an existing edge does nothing. The instruction does not change the
    quantum state or make an unsupported gate available; it only satisfies the
    connectivity requirement for a supported two-atom gate.

    ``Pair`` takes exactly two distinct scalar targets and must be
    unconditional. The atom-array simulator rejects a condition when the
    program runs. Attached `fatqat.noise.Loss` or supported finite channel
    noise still acts on the targets. `fatqat.RegisterView` targets are rejected.

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

    ``Unpair`` is implemented only by `fatqat.simulator.AtomArraySimulator`;
    other built-in backends raise `fatqat.errors.UnsupportedOperationError`.
    Targets are the two distinct atoms ``(a, b)``; their order does not affect
    connectivity.
    Removing a missing edge does nothing. The instruction does not change the
    quantum state, but attached `fatqat.noise.Loss` or supported finite channel
    noise still acts on its targets.

    ``Unpair`` takes exactly two distinct scalar targets and must be
    unconditional. The atom-array simulator rejects a condition when the
    program runs. After unpairing, a supported two-atom gate on that pair fails
    its connectivity check until another ``Pair``. `fatqat.RegisterView`
    targets are rejected.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.Unpair, (0, 1))
    """

    name: ClassVar[str] = "Unpair"
    num_subsystems: ClassVar[int] = 2


Pair = PairGate()
Unpair = UnpairGate()
