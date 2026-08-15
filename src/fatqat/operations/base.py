"""Operation base class shared by every gate and instruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..registers import RegisterRef


@dataclass(frozen=True)
class Operation:
    """Base class for immutable operation objects.

    Fixed gates are exposed as pre-built singleton values in
    ``fatqat.operations`` (normally imported as ``op``).
    Parametric gates are exposed as classes and should be instantiated, such as
    ``RX(theta)``.

    Attributes:
        name: Public operation name.
        _num_subsystems: Number of quantum targets required by the operation, or
            None for variable arity governed by ``min_targets``. A minimum of
            zero supports global operations whose target set is implicit.

    Examples:
        >>> import fatqat.operations as op
        >>> op.H.name
        'H'
        >>> op.H.num_targets
        1
        >>> op.CX.num_targets
        2
        >>> op.RX(0.2).num_targets
        1
    """

    name: ClassVar[str] = "OP"
    _num_subsystems: ClassVar[int | None] = 1
    _min_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = False
    """Whether this operation accepts a ``RegisterView`` target expression in
    addition to scalar ``RegisterRef`` targets. Only RX, RY, RZ, CX, and CZ
    opt in (set ``True``); every other operation stays scalar-only. This is
    the single, centralized capability flag consulted by
    ``AppliedOperation.__post_init__`` -- new code should read
    ``accepts_views`` rather than checking operation identity or name.
    """
    _is_direct_control: ClassVar[bool] = False
    """Whether the operation is a direct physical-control block.

    Direct controls have zero ordinary target arity but must not be registered
    as calibrated gates or operation-scoped gate-noise selectors.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        # Validate the arity class constant once, at class-definition time,
        # where a bad value is actually a developer error - rather than on
        # every instantiation of an already-correct class.
        super().__init_subclass__(**kwargs)
        n = cls._num_subsystems
        minimum = cls._min_subsystems
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise ValueError(
                f"_min_subsystems must be a non-negative int, got {minimum!r}"
            )
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
            raise ValueError(
                f"_num_subsystems must be a non-negative int or None, got {n!r}"
            )

    @property
    def num_targets(self) -> int | None:
        """Number of quantum targets required, or None for variable arity."""
        return type(self)._num_subsystems

    @property
    def min_targets(self) -> int:
        """Minimum accepted targets, or the exact arity for a fixed operation."""
        fixed = self.num_targets
        return type(self)._min_subsystems if fixed is None else fixed

    @property
    def accepts_views(self) -> bool:
        """Whether this operation accepts a ``RegisterView`` target expression."""
        return type(self)._accepts_views

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Raise ValueError if this operation's parameters are invalid for the
        resolved target references. Default no-op; override for gates whose
        parameters name basis levels (or otherwise depend on target identity
        or dimension). Reads dimension as targets[i].register.dim, consistent
        with the matrix-rule contract.
        """
        return
