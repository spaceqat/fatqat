"""Register / RegisterRef value objects (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, eq=False)
class Register:
    """Base class for a fixed-size resource register.

    Use ``QuantumRegister`` or ``ClassicalRegister`` for registers passed to a
    ``Program``; a plain ``Register`` does not identify either resource kind.
    Indexing creates an immutable ``RegisterRef``.

    A register is a frozen identity object. Two separately constructed
    registers remain distinct even when every field has the same value, and
    ``name`` does not need to be unique. ``metadata`` is shallow-copied into a
    new mutable dictionary: later top-level changes to the input are not
    observed, but nested values remain shared.

    Args:
        size: Positive number of slots. Must be an integer, not ``bool``.
        name: Optional display label. It does not determine register identity.
        metadata: Mapping with string keys and arbitrary application values
            (default empty). FATQAT shallow-copies the mapping and does not
            inspect or reserve any key.
        dim: Local dimension of every slot. ``2`` (the default) represents a
            qubit or bit; values greater than ``2`` represent qudits or
            d-ary classical digits. Must be an integer, not ``bool``.

    Raises:
        TypeError: If ``size`` or ``dim`` is not an integer, or ``metadata``
            cannot be copied into a dictionary.
        ValueError: If ``size`` is not positive or ``dim`` is less than ``2``.

    Examples:
        Index a concrete register to get a ``RegisterRef``:

        >>> import fatqat as fq
        >>> qreg = fq.QuantumRegister(2, name="q")
        >>> qreg[0]
        RegisterRef(register=QuantumRegister(size=2, name='q', metadata={}, dim=2), index=0)
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

        Each call creates an equal immutable reference for the same register
        object and index. Python-style negative indexing is not supported.

        Args:
            index: Zero-based slot index. Must be an integer, not ``bool``.

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
    """Fixed-size register whose refs name quantum operation targets.

    It follows ``Register`` identity and metadata ownership rules.
    Operation and backend support for a local dimension is checked separately
    when a program is built or run.

    Args:
        size: Positive number of quantum slots. Must be an integer, not
            ``bool``.
        name: Optional display label; it does not determine register identity.
        metadata: String-keyed application metadata (default empty),
            shallow-copied into a mutable dictionary.
        dim: Local dimension of every slot. ``2`` (the default) creates
            qubits; a larger value creates qudits. Must be an integer, not
            ``bool``.

    Raises:
        TypeError: If ``size`` or ``dim`` is not an integer, or ``metadata``
            cannot be copied into a dictionary.
        ValueError: If ``size`` is not positive or ``dim`` is less than ``2``.
    """


@dataclass(frozen=True, eq=False)
class ClassicalRegister(Register):
    """Fixed-size register for measurement results and conditions.

    A slot stores a digit from ``0`` through ``dim - 1``. Measuring a quantum
    slot into a classical slot requires equal dimensions. It follows
    ``Register`` identity and metadata ownership rules.

    Args:
        size: Positive number of classical slots. Must be an integer, not
            ``bool``.
        name: Optional display label; it does not determine register identity.
        metadata: String-keyed application metadata (default empty),
            shallow-copied into a mutable dictionary.
        dim: Number of possible values in each slot. ``2`` (the default)
            creates bits; a larger value creates d-ary digits. Must be an
            integer, not ``bool``.

    Raises:
        TypeError: If ``size`` or ``dim`` is not an integer, or ``metadata``
            cannot be copied into a dictionary.
        ValueError: If ``size`` is not positive or ``dim`` is less than ``2``.
    """


@dataclass(frozen=True)
class RegisterRef:
    """Immutable, hashable reference to one slot in a register.

    Obtain refs by indexing a register. Equality combines the register's
    identity with the index, so refs from lookalike register objects remain
    distinct and can safely be dictionary keys.

    Args:
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
    """Immutable value selecting every grid member in row-major order.

    Normally created by ``GridRegister.all`` rather than directly.
    """


@dataclass(frozen=True)
class RowSelector:
    """Immutable value selecting one row in increasing column order.

    Args:
        row: Zero-based row index.
    """

    row: int


@dataclass(frozen=True)
class ColumnSelector:
    """Immutable value selecting one column in increasing row order.

    Args:
        col: Zero-based column index.
    """

    col: int


@dataclass(frozen=True)
class BlockSelector:
    """Immutable value selecting a rectangular sub-block of a grid.

    Members are in row-major order inside the selected rectangle. Ranges are
    zero-based and half-open.

    Args:
        rows: ``(start, stop)`` row range.
        cols: ``(start, stop)`` column range.
    """

    rows: tuple[int, int]
    cols: tuple[int, int]


Selector = AllSelector | RowSelector | ColumnSelector | BlockSelector


@dataclass(frozen=True, eq=False, init=False)
class GridRegister(QuantumRegister):
    """Quantum register arranged as a rectangular, row-major grid.

    The flat slot at coordinate ``(row, col)`` has index
    ``row * cols + col``. The grid is a backend-neutral program resource: its
    shape does not assign physical sites or guarantee that a backend preserves
    the coordinates.

    ``all``, ``row``, ``column``, and ``block`` create immutable
    ``RegisterView`` targets in deterministic member order.

    Like other registers, a grid register is frozen and compares by identity.
    ``metadata`` is shallow-copied into a new, still-mutable dictionary;
    nested values are not copied.

    Args:
        rows: Positive number of rows. Must be an integer, not ``bool``.
        cols: Positive number of columns. Must be an integer, not ``bool``.
        name: Optional display label; it does not determine register identity.
        metadata: String-keyed application metadata, or ``None`` for an empty
            mapping. FATQAT shallow-copies the mapping into a mutable
            dictionary.
        dim: Local dimension of every grid member. ``2`` (the default)
            creates qubits; a larger value creates qudits. Must be an integer,
            not ``bool``.

    Raises:
        TypeError: If ``rows``, ``cols``, or ``dim`` is not an integer, or
            ``metadata`` cannot be copied into a dictionary.
        ValueError: If ``rows`` or ``cols`` is not positive or ``dim`` is less
            than ``2``.

    Examples:
        >>> import fatqat as fq
        >>> qubits = fq.GridRegister(2, 3, name="qubits")
        >>> qubits.size
        6
        >>> qubits[4]
        RegisterRef(register=GridRegister(size=6, name='qubits', metadata={}, dim=2, rows=2, cols=3), index=4)
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
        """Return a view selecting every member in row-major order.

        Returns:
            An immutable view bound to this register.
        """
        return RegisterView(register=self, selector=AllSelector())

    def row(self, row: int) -> "RegisterView":
        """Return a view selecting one row, in increasing column order.

        Args:
            row: Zero-based row index. Must be an integer, not ``bool``.

        Returns:
            An immutable view bound to this register.

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

        Args:
            col: Zero-based column index. Must be an integer, not ``bool``.

        Returns:
            An immutable view bound to this register.

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
            rows: Zero-based, half-open ``(start, stop)`` row range. Endpoints
                must be integers, not ``bool``.
            cols: Zero-based, half-open ``(start, stop)`` column range.
                Endpoints must be integers, not ``bool``.

        Returns:
            An immutable view bound to this register.

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
    """Immutable, hashable target selecting members of one grid register.

    Create views with ``GridRegister.all``, ``row``, ``column``, or ``block``
    rather than calling this class directly; its selector representation is
    not a public construction contract. The ``register`` attribute identifies
    the owning grid. Views compare and hash by that register's identity and
    their selection. A unary view-capable operation is applied once to each
    selected member. A two-target view operation pairs the two views member by
    member. The built-in view-capable operations are ``RX``, ``RY``, ``RZ``,
    ``CX``, and ``CZ``; all other built-ins require scalar targets.

    ``Program.add`` requires the view's grid to belong to that program. For a
    two-target view operation, both operands must be views of the same
    selector kind and cardinality; views on the same grid must not overlap.
    Invalid combinations raise ``ValueError`` when the operation is added.

    Member order is deterministic: ``all()`` and ``block()`` use row-major
    order, ``row()`` uses increasing column order, and ``column()`` uses
    increasing row order.

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
