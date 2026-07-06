"""Operation base class shared by every gate and instruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..registers import RegisterRef


@dataclass(frozen=True)
class Operation:
    """Base class for immutable operation objects.

    Fixed gates are exposed as pre-built singleton values in ``fatqat.ops``.
    Parametric gates are exposed as classes and should be instantiated, such as
    ``RX(theta)``.

    Attributes:
        name: Public operation name.
        _num_subsystems: Number of quantum targets required by the operation, or
            None for variable arity with at least one target.
    """

    name: ClassVar[str] = "OP"
    _num_subsystems: ClassVar[int | None] = 1

    def __init_subclass__(cls, **kwargs) -> None:
        # Validate the arity class constant once, at class-definition time,
        # where a bad value is actually a developer error - rather than on
        # every instantiation of an already-correct class.
        super().__init_subclass__(**kwargs)
        n = cls._num_subsystems
        if n is None:
            return
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_subsystems must be a positive int or None, got {n!r}")

    @property
    def num_subsystems(self) -> int | None:
        """Number of quantum targets required, or None for variable arity."""
        return type(self)._num_subsystems

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Raise ValueError if this operation's parameters are invalid for the
        resolved target references. Default no-op; override for gates whose
        parameters name basis levels (or otherwise depend on target identity
        or dimension). Reads dimension as targets[i].register.dim, consistent
        with the matrix-rule contract.
        """
        return
