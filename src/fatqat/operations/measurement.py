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
    re-validate ``targets``/``outputs`` element types or tuple-ness:
    ``add_measurement`` already guarantees well-formed ``RegisterRef`` tuples
    of the right register kind via
    ``_resolve_quantum_ref``/``_resolve_classical_ref``.
    Constructing this class directly skips that guarantee - see
    ``AppliedOperation`` for the same tradeoff and its consequences. It does,
    however, own the structural invariants that hold for any well-typed refs -
    equal length, non-empty, and per-pair quantum/classical dimension match -
    so those are checked once here and not duplicated in
    ``add_measurement``/``measure_all``.

    Attributes:
        targets: Quantum refs to measure, stored as a tuple. Named to match
            ``AppliedOperation.targets``: these are the subsystems the
            instruction acts on, not the registers holding them.
        outputs: Classical refs the outcomes are written into, stored as a
            tuple. Not "clbits" - a `ClassicalRegister` carries ``dim``, so a
            slot holds a d-ary digit.

    Examples:
        >>> import fatqat as fq
        >>> program = fq.Program(1, 1)
        >>> m = fq.ops.Measurement(
        ...     targets=(program.qreg[0][0],), outputs=(program.clreg[0][0],)
        ... )
        >>> m.targets
        (RegisterRef(register=QuantumRegister(size=1, name='q', metadata={}, dim=2), index=0),)
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
