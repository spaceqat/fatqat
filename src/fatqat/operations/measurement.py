"""Measurement instruction: a computational-basis readout into classical slots."""

from __future__ import annotations

from dataclasses import dataclass

from ..registers import RegisterRef


@dataclass(frozen=True)
class Measurement:
    """Record a computational-basis measurement into classical slots.

    Measurements are instruction records with frozen fields and retain
    insertion order alongside operations. Each quantum target is paired with
    the classical output at the same tuple position. Their local dimensions
    must match, so a qudit outcome is written to a classical slot of the same
    dimension.

    Prefer ``Program.measure`` or ``Program.measure_all`` to construct these
    values. Those methods resolve integers, verify register kind and ownership,
    and pass tuples here. The direct constructor stores ``targets`` and
    ``outputs`` exactly as supplied: it does not copy them or validate that
    they are tuples of the correct ``RegisterRef`` kind. It only checks the
    structural invariants described below, so malformed direct input may fail
    later during backend preparation. A mutable container therefore remains
    mutable and can make the record unhashable despite its frozen fields.

    Repeated targets and outputs are accepted. Built-in backends report every
    target/output pair in tuple order; a repeated target repeats its collapsed
    physical outcome (with reporting noise resolved per pair), and a repeated
    classical output retains the later pair's reported value.

    Args:
        targets: Non-empty tuple of quantum references to measure.
        outputs: Tuple of classical references receiving the corresponding
            outcomes. It must have the same length and per-position dimensions
            as ``targets``.

    Raises:
        ValueError: If the tuples are empty, have different lengths, or a
            target/output pair has different local dimensions.

    Examples:
        >>> import fatqat as fq
        >>> program = fq.Program(1, 1)
        >>> program.measure(0, 0)
    """

    targets: tuple[RegisterRef, ...]
    outputs: tuple[RegisterRef, ...]

    def __post_init__(self) -> None:
        if len(self.targets) != len(self.outputs):
            raise ValueError(
                "measurement targets and outputs must have the same number of entries"
            )
        if len(self.targets) < 1:
            raise ValueError("measurement requires at least one target/output pair")
        for pos, (q, c) in enumerate(zip(self.targets, self.outputs)):
            if q.register.dim != c.register.dim:
                raise ValueError(
                    f"measurement operand {pos}: quantum dim {q.register.dim} "
                    f"!= classical dim {c.register.dim}"
                )
