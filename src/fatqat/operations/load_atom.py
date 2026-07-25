"""LoadAtom: atom-grid-specific site-loading instruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation


@dataclass(frozen=True)
class LoadAtom(Operation):
    """Declares that a top-left `rows x cols` block of device sites now holds atoms.

    Zero-arity: `LoadAtom` carries no quantum targets, since neutral-atom
    loading is a per-device-site fact, not a per-qubit gate
    (``program.add(ops.LoadAtom(2, 3))`` needs no target operand). It has no
    matrix implementation and is recognized by type during lowering, exactly
    like `~fatqat.ops.Barrier`/`~fatqat.ops.Reset`; a backend with no special
    handling for it rejects it as an unsupported operation.

    `~fatqat.backends.FakeAtomGridBackend` is currently the only backend that
    interprets it: it requires `LoadAtom` to be a program's first
    instruction, unconditional, and to appear at most once.

    Attributes:
        rows: Number of loaded rows, counted from the device's top-left corner.
        cols: Number of loaded columns, counted from the device's top-left corner.

    Examples:
        >>> import fatqat as fq
        >>> program = fq.Program(4)
        >>> program.add(fq.ops.LoadAtom(2, 2))
        >>> program.operations[0].targets
        ()
    """

    rows: int
    cols: int
    name: ClassVar[str] = "LoadAtom"
    _num_subsystems: ClassVar[int] = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rows, int) or isinstance(self.rows, bool):
            raise TypeError(f"rows must be int, got {type(self.rows)!r}")
        if self.rows <= 0:
            raise ValueError(f"rows must be positive, got {self.rows}")
        if not isinstance(self.cols, int) or isinstance(self.cols, bool):
            raise TypeError(f"cols must be int, got {type(self.cols)!r}")
        if self.cols <= 0:
            raise ValueError(f"cols must be positive, got {self.cols}")
