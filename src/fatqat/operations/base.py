"""Operation base class shared by every gate and instruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..registers import RegisterRef


@dataclass(frozen=True)
class Operation:
    """Base value for operations accepted by `fatqat.Program.add`.

    Parameter-free gates and structural operations are ready-to-use values such
    as ``ops.H`` and ``ops.Reset``; do not call them. Construct parameterized
    values such as ``ops.RX(0.2)`` before adding them.

    ``Operation`` is also the extension base for custom operations. An
    ordinary subclass declares a public ``name`` and ``num_subsystems``. A
    non-negative integer gives the exact number of logical target
    expressions; ``None`` gives variable arity with a minimum of one target.
    A custom operation also needs a compatible backend implementation;
    subclassing alone does not make it executable.

    Attributes:
        name: Stable name used in diagnostics and user-facing displays.
        num_subsystems: Exact non-negative number of logical targets, or
            ``None`` for variable arity.

    Raises:
        ValueError: At subclass definition if ``num_subsystems`` is negative,
            boolean, or not an integer or ``None``.

    Examples:
        >>> import fatqat.operations as ops
        >>> (ops.H.num_subsystems, ops.CX.num_subsystems)
        (1, 2)
        >>> ops.RX(0.2).name
        'RX'
    """

    name: ClassVar[str] = "OP"
    num_subsystems: ClassVar[int | None] = 1
    _min_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = False
    """Whether this operation accepts a ``RegisterView`` target expression in
    addition to scalar ``RegisterRef`` targets.
    """
    _is_direct_control: ClassVar[bool] = False
    """Whether the operation is a direct physical-control block."""

    def __init_subclass__(cls, **kwargs) -> None:
        # Validate the arity class constant once, at class-definition time,
        # where a bad value is actually a developer error - rather than on
        # every instantiation of an already-correct class.
        super().__init_subclass__(**kwargs)
        for retired_name in ("num_targets", "_num_subsystems"):
            if retired_name in cls.__dict__:
                raise TypeError(
                    f"{retired_name} is no longer supported on Operation "
                    "subclasses; declare num_subsystems instead"
                )
        n = cls.num_subsystems
        minimum = cls._min_subsystems
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise ValueError(
                f"_min_subsystems must be a non-negative int, got {minimum!r}"
            )
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
            raise ValueError(
                f"num_subsystems must be a non-negative int or None, got {n!r}"
            )

    @property
    def min_targets(self) -> int:
        """Return the minimum accepted target count.

        This is the exact count for fixed-arity operations. Public built-in
        variable-arity operations require at least one target.
        """
        fixed = self.num_subsystems
        return type(self)._min_subsystems if fixed is None else fixed

    @property
    def accepts_views(self) -> bool:
        """Return whether `fatqat.Program.add` accepts `fatqat.RegisterView` targets.

        The built-in view-capable operations are `RX`, `RY`, `RZ`, `CX`, and
        `CZ`. All other built-ins require scalar targets. A
        custom unary or two-target subclass may override this property to opt
        into memberwise view application.
        """
        return type(self)._accepts_views

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Validate parameters against resolved scalar targets.

        The base implementation accepts every target tuple. Override this hook
        when an operation parameter depends on a target property such as local
        dimension. Scalar-target errors arise from `fatqat.Program.add`; for a
        `fatqat.RegisterView`, the hook is applied to each selected member when
        the program runs.

        Args:
            targets: Resolved scalar `fatqat.RegisterRef` values in operand order.

        Raises:
            ValueError: If the operation's parameters are incompatible with
                the targets.
        """
        return
