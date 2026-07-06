"""Measurement instruction: a computational-basis readout into classical slots."""

from __future__ import annotations

from dataclasses import dataclass

from ..registers import RegisterRef


@dataclass(frozen=True)
class Measurement:
    """A measurement from one or more quantum refs into matching classical slots.

    Measurements live in ``Program.operations`` alongside applied operations
    and preserve insertion order.

    Like ``AppliedOperation``, ``__post_init__`` intentionally does not
    re-validate ``qreg``/``clreg`` element types or tuple-ness:
    ``add_measurement`` already guarantees well-formed ``RegisterRef`` tuples
    of the right register kind via ``_resolve_qubit``/``_resolve_clbit``.
    Constructing this class directly skips that guarantee - see
    ``AppliedOperation`` for the same tradeoff and its consequences. It does,
    however, own the structural invariants that hold for any well-typed refs -
    equal length, non-empty, and per-pair quantum/classical dimension match -
    so those are checked once here and not duplicated in
    ``add_measurement``/``measure_all``.

    Attributes:
        qreg: Quantum register references to measure, stored as a tuple.
        clreg: Classical register references to write, stored as a tuple.
    """

    qreg: tuple[RegisterRef, ...]
    clreg: tuple[RegisterRef, ...]

    def __post_init__(self) -> None:
        if len(self.qreg) != len(self.clreg):
            raise ValueError("measurement qreg and clreg must have the same number of entries")
        if len(self.qreg) < 1:
            raise ValueError("measurement requires at least one qreg/clreg pair")
        for pos, (q, c) in enumerate(zip(self.qreg, self.clreg)):
            if q.register.dim != c.register.dim:
                raise ValueError(
                    f"measurement operand {pos}: quantum dim {q.register.dim} "
                    f"!= classical dim {c.register.dim}"
                )
