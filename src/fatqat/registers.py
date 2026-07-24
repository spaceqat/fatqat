"""Register / RegisterRef value objects (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, eq=False)
class Register:
    """Base value object for a fixed-size resource register.

    Register metadata is copied at construction time, and register objects are
    frozen. Use indexing to create ``RegisterRef`` values.

    Attributes:
        size: Number of slots in the register. Must be a positive integer.
        name: Optional user-facing register name.
        metadata: User metadata copied into the register.
        dim: Dimension of each slot (default 2 for qubits). Must be an integer >= 2.

    Examples:
        Index into a register to get a ``RegisterRef``:

        >>> import fatqat as fq
        >>> qreg = fq.QuantumRegister(2, name="q")
        >>> qreg[0]
        RegisterRef(register=QuantumRegister(size=2, name='q', metadata={}, dim=2), index=0)
        >>> qreg[5]
        Traceback (most recent call last):
            ...
        IndexError: 5

        A qutrit register uses ``dim=3``:

        >>> qutrit = fq.QuantumRegister(1, dim=3)
        >>> qutrit.dim
        3
    """

    size: int
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dim: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise TypeError(f"register size must be int, got {type(self.size)!r}")
        if self.size <= 0:
            raise ValueError(f"register size must be positive, got {self.size}")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError(f"register dim must be int, got {type(self.dim)!r}")
        if self.dim < 2:
            raise ValueError(
                f"register dim must be >= 2 (a nontrivial system), got {self.dim}"
            )
        # Copy metadata so a caller's later mutation can't reach into this frozen
        # value object.
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __getitem__(self, index: int) -> "RegisterRef":
        """Return a reference to one slot in this register.

        Args:
            index: Zero-based slot index. Negative indexing is not supported.

        Returns:
            A ``RegisterRef`` pointing at this register and index.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside ``0 <= index < size``.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"register index must be int, got {type(index)!r}")
        if not 0 <= index < self.size:
            raise IndexError(index)
        return RegisterRef(register=self, index=index)


@dataclass(frozen=True, eq=False)
class QuantumRegister(Register):
    """Register whose refs may be used as quantum operation targets."""


@dataclass(frozen=True, eq=False)
class ClassicalRegister(Register):
    """Register whose refs may receive measurement results and conditions."""


@dataclass(frozen=True)
class RegisterRef:
    """Reference to one slot in a register.

    Attributes:
        register: Register object being referenced.
        index: Zero-based slot index within ``register``.
    """

    register: Register
    index: int


def _validate_range(value: Any, limit: int, label: str) -> tuple[int, int]:
    """Validate a zero-based, half-open ``(start, stop)`` range.

    Args:
        value: Candidate ``(start, stop)`` pair.
        limit: Exclusive upper bound that ``stop`` must not exceed.
        label: Human-readable name used in error messages.

    Returns:
        The validated ``(start, stop)`` pair.

    Raises:
        TypeError: If ``value`` is not a two-element pair of ints.
        ValueError: If the range does not satisfy ``0 <= start < stop <= limit``.
    """
    try:
        start, stop = value
    except (TypeError, ValueError):
        raise TypeError(
            f"{label} must be a (start, stop) pair, got {value!r}"
        ) from None
    for part_name, part_value in (("start", start), ("stop", stop)):
        if not isinstance(part_value, int) or isinstance(part_value, bool):
            raise TypeError(
                f"{label} {part_name} must be int, got {type(part_value)!r}"
            )
    if not 0 <= start < stop <= limit:
        raise ValueError(
            f"{label} range must satisfy 0 <= start < stop <= {limit}, "
            f"got ({start}, {stop})"
        )
    return start, stop


@dataclass(frozen=True)
class AllSelector:
    """Selects every member of a ``GridRegister``, in row-major order."""


@dataclass(frozen=True)
class RowSelector:
    """Selects one row of a ``GridRegister``, in increasing column order.

    Attributes:
        row: Zero-based row index.
    """

    row: int


@dataclass(frozen=True)
class ColumnSelector:
    """Selects one column of a ``GridRegister``, in increasing row order.

    Attributes:
        col: Zero-based column index.
    """

    col: int


@dataclass(frozen=True)
class BlockSelector:
    """Selects a rectangular sub-block of a ``GridRegister``.

    Members are in row-major order inside the selected rectangle. Ranges are
    zero-based and half-open.

    Attributes:
        rows: ``(start, stop)`` row range.
        cols: ``(start, stop)`` column range.
    """

    rows: tuple[int, int]
    cols: tuple[int, int]


Selector = AllSelector | RowSelector | ColumnSelector | BlockSelector


@dataclass(frozen=True, eq=False, init=False)
class GridRegister(QuantumRegister):
    """Quantum register arranged as a rectangular, row-major grid.

    A ``GridRegister`` is a backend-neutral logical resource: it carries
    ``rows`` and ``cols`` but no physical-site or placement information.
    Selection helpers (``all``, ``row``, ``column``, ``block``) return
    immutable ``RegisterView`` values that hold a structured selector rather
    than an eagerly expanded tuple of refs.

    Like ``Register``/``QuantumRegister``, ``GridRegister`` uses identity
    equality and hashing: two grid registers built with identical arguments
    are distinct registers.

    Attributes:
        rows: Number of rows. Must be a positive integer.
        cols: Number of columns. Must be a positive integer.

    Examples:
        >>> import fatqat as fq
        >>> atoms = fq.GridRegister(2, 3, name="atoms")
        >>> atoms.size
        6
        >>> atoms[4]
        RegisterRef(register=GridRegister(size=6, name='atoms', metadata={}, dim=2, rows=2, cols=3), index=4)
    """

    rows: int
    cols: int

    def __init__(
        self,
        rows: int,
        cols: int,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dim: int = 2,
    ) -> None:
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise TypeError(f"rows must be int, got {type(rows)!r}")
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        if not isinstance(cols, int) or isinstance(cols, bool):
            raise TypeError(f"cols must be int, got {type(cols)!r}")
        if cols <= 0:
            raise ValueError(f"cols must be positive, got {cols}")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "size", rows * cols)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})
        object.__setattr__(self, "dim", dim)
        # Reuses Register.__post_init__'s size/dim validation and defensive
        # metadata copy.
        self.__post_init__()

    def all(self) -> "RegisterView":
        """Return a view selecting every member, in row-major order."""
        return RegisterView(register=self, selector=AllSelector())

    def row(self, row: int) -> "RegisterView":
        """Return a view selecting one row, in increasing column order.

        Raises:
            TypeError: If ``row`` is not an integer.
            IndexError: If ``row`` is outside ``0 <= row < self.rows``.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be int, got {type(row)!r}")
        if not 0 <= row < self.rows:
            raise IndexError(row)
        return RegisterView(register=self, selector=RowSelector(row=row))

    def column(self, col: int) -> "RegisterView":
        """Return a view selecting one column, in increasing row order.

        Raises:
            TypeError: If ``col`` is not an integer.
            IndexError: If ``col`` is outside ``0 <= col < self.cols``.
        """
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be int, got {type(col)!r}")
        if not 0 <= col < self.cols:
            raise IndexError(col)
        return RegisterView(register=self, selector=ColumnSelector(col=col))

    def block(self, rows: tuple[int, int], cols: tuple[int, int]) -> "RegisterView":
        """Return a view selecting a rectangular sub-block.

        Members are in row-major order inside the selected rectangle.

        Args:
            rows: Zero-based, half-open ``(start, stop)`` row range.
            cols: Zero-based, half-open ``(start, stop)`` column range.

        Raises:
            TypeError: If either range is not a two-element pair of ints.
            ValueError: If either range does not satisfy
                ``0 <= start < stop <= limit`` for the register's row/column
                count.
        """
        row_range = _validate_range(rows, self.rows, "rows")
        col_range = _validate_range(cols, self.cols, "cols")
        return RegisterView(
            register=self,
            selector=BlockSelector(rows=row_range, cols=col_range),
        )


@dataclass(frozen=True)
class RegisterView:
    """Immutable, hashable structured target expression over one ``GridRegister``.

    A view is bound to exactly one ``GridRegister`` (compared by that
    register's identity, matching ``GridRegister``'s own identity semantics)
    and stores a structured ``selector`` instead of an eagerly expanded tuple
    of refs. This preserves the correct operand form for later reshape-aware
    binding, which resolves a view against the register's arrangement at that
    point in the instruction sequence.

    Member order, once a view is resolved, is deterministic:

    - ``AllSelector``: row-major;
    - ``RowSelector``: increasing column;
    - ``ColumnSelector``: increasing row;
    - ``BlockSelector``: row-major inside the selected rectangle.

    Attributes:
        register: The ``GridRegister`` this view is bound to.
        selector: The structured selector (``AllSelector``, ``RowSelector``,
            ``ColumnSelector``, or ``BlockSelector``).
    """

    register: GridRegister
    selector: Selector


def _view_members(view: RegisterView) -> tuple[RegisterRef, ...]:
    """Enumerate a view in its documented deterministic member order."""
    register = view.register
    selector = view.selector
    if isinstance(selector, AllSelector):
        coordinates = [
            (row, col) for row in range(register.rows) for col in range(register.cols)
        ]
    elif isinstance(selector, RowSelector):
        coordinates = [(selector.row, col) for col in range(register.cols)]
    elif isinstance(selector, ColumnSelector):
        coordinates = [(row, selector.col) for row in range(register.rows)]
    elif isinstance(selector, BlockSelector):
        (row_start, row_stop), (col_start, col_stop) = selector.rows, selector.cols
        coordinates = [
            (row, col)
            for row in range(row_start, row_stop)
            for col in range(col_start, col_stop)
        ]
    else:  # pragma: no cover - exhaustive over the Selector union
        raise AssertionError(f"unhandled selector {selector!r}")
    return tuple(register[row * register.cols + col] for row, col in coordinates)


def _views_overlap(first: RegisterView, second: RegisterView) -> bool:
    """Whether two same-register, same-selector-type views can share a member.

    Callers must have already established both preconditions; this only
    decides the overlap question. Rows and columns each partition the grid,
    so two unequal row selectors (or column selectors) are always fully
    disjoint - no partial-overlap case exists for them. Blocks have no such
    guarantee: two distinct, unequal blocks can still share cells, so they
    are checked by row/col range intersection instead of equality.
    """
    selector, other = first.selector, second.selector
    if isinstance(selector, AllSelector):
        return True  # only one possible AllSelector value: always identical
    if isinstance(selector, RowSelector):
        return selector.row == other.row
    if isinstance(selector, ColumnSelector):
        return selector.col == other.col
    if isinstance(selector, BlockSelector):
        row_overlap = (
            selector.rows[0] < other.rows[1] and other.rows[0] < selector.rows[1]
        )
        col_overlap = (
            selector.cols[0] < other.cols[1] and other.cols[0] < selector.cols[1]
        )
        return row_overlap and col_overlap
    raise AssertionError(f"unhandled selector {selector!r}")  # pragma: no cover


def _validate_view_pair(
    first: RegisterView, second: RegisterView, *, op_name: str
) -> None:
    """Validate that two views may legally pair as one two-target operation's operands.

    Cross-selector-type pairing is disallowed outright - only row/row,
    column/column, block/block, and all/all pairs are legal - since a mixed
    pair (e.g. a row and a column) can partially overlap in ways neither
    selector alone expresses. Same-register pairs must additionally not
    overlap (see `_views_overlap`); views on different registers can never
    overlap (a `RegisterRef`'s identity includes its register) but must
    still match cardinality, since pairwise application zips them together.
    """
    if type(first.selector) is not type(second.selector):
        raise ValueError(
            f"{op_name} pairs a {type(first.selector).__name__} view with a "
            f"{type(second.selector).__name__} view; grouped two-target "
            "application requires both views to use the same selector kind"
        )
    first_size, second_size = len(_view_members(first)), len(_view_members(second))
    if first_size != second_size:
        raise ValueError(
            f"{op_name} pairs views of unequal size "
            f"({first_size} vs {second_size}); pairwise view application "
            "requires equal cardinality"
        )
    if first.register is second.register and _views_overlap(first, second):
        raise ValueError(
            f"{op_name} pairs overlapping views on the same register; a "
            "member cannot be both control and target of one two-target "
            "application"
        )
