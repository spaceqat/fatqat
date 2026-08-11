"""Rearrange: atom-grid mid-circuit relabeling of atoms to new device sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..registers import RegisterRef
from .base import Operation


@dataclass(frozen=True)
class Rearrange(Operation):
    """Move the operand atoms to new device sites mid-circuit.

    The operands (``Program.add`` targets) are the atoms to move; ``sites``
    gives each one's destination device-site label, aligned by position:
    target ``i`` moves to ``sites[i]``.

    Rearrange changes only where an atom sits, never the quantum state
    (M-B2): the backend relabels the resource layout and emits no execution
    step, so the same statevector runs before and after. Its point is to make
    a two-qubit gate legal on a pair that started non-adjacent, by bringing
    them onto nearest-neighbor sites -- the grid-adjacency stand-in for the
    Rydberg blockade, not a continuous distance model, with no limit on how
    far an atom moves in one step. It is atomic: exchanging two atoms
    (``{a: site_b, b: site_a}``) is legal and needs no temporary site (S-B1).
    It must be unconditional; a conditional Rearrange is rejected at lowering
    (M-B6).

    This is not the SWAP gate: ``op.Swap`` exchanges two qubits' quantum
    states (a unitary), whereas a Rearrange swap exchanges their trap sites
    and leaves the state untouched -- each atom carries its state to the new
    trap.

    ``sites`` carries destinations rather than the refs because ``Program.add``
    already binds the refs as this operation's targets -- which is what lets
    channel noise attach to the moved atoms.

    Attributes:
        sites: Destination device-site labels, one per target, in order.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as op
        >>> atoms = fq.GridRegister(1, 2, name="atoms")
        >>> program = fq.Program([atoms])
        >>> program.add(op.LoadAtoms(1, 2))
        >>> program.add(op.Rearrange((1, 0)), (atoms[0], atoms[1]))  # swap the two atoms
        >>> program.operations[1].operation.sites
        (1, 0)
    """

    sites: tuple[int, ...]
    name: ClassVar[str] = "Rearrange"
    _num_subsystems: ClassVar[int | None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.sites, tuple):
            object.__setattr__(self, "sites", tuple(self.sites))
        for site in self.sites:
            if not isinstance(site, int) or isinstance(site, bool):
                raise TypeError(f"Rearrange site must be int, got {type(site)!r}")
            if site < 0:
                raise ValueError(f"Rearrange site must be non-negative, got {site}")

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Reject a target count that does not match the number of sites."""
        if len(targets) != len(self.sites):
            raise ValueError(
                f"Rearrange has {len(self.sites)} site(s) but was applied to "
                f"{len(targets)} target(s)"
            )