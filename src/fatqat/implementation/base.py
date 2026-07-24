"""Matrix-implementation abstraction: the rule protocol, its wrappers, and the
implementation map (unconstrained and device-specific).

A matrix implementation maps an operation to its local matrix (physics only).
The backend pairs that matrix with layout-resolved target indices to build an
``ApplyMatrixStep`` (see ``backends.steps``), the plain data container the
statevector engine reads directly.

Local matrix convention (binding for every entry in this package):
    - ``AppliedOperation.targets`` operand order defines the local
      tensor-factor order; ``targets[0]`` is the local most-significant bit
      (MSB), ``targets[-1]`` the local least-significant bit (LSB). See
      ``engine._apply_matrix`` for the little-endian contraction this feeds.
    - For every controlled gate (``CX``, ``CZ``, ``CY``, and the controlled
      gates added in later batches), the control operand(s) come first and the
      target operand(s) come last: operand 0 (and operand 1 for
      doubly-controlled gates) is the control, occupying the local MSB
      position(s).

A matrix implementation rule receives the bare :py:class:`~fatqat.operations.Operation` instance that was
applied (e.g. `RX(0.3)`) plus the `targets: tuple[RegisterRef, ...]` operand
tuple by keyword, and returns the local matrix, never the surrounding
`AppliedOperation`, and never a feedforward `condition`: condition resolution
happens separately, in the backend. `targets` lets a rule read
`targets[0].register.dim` to build a dimension-dependent matrix (e.g. a
qudit `Shift`/`Clock`/`Sum` gate); a rule whose matrix never depends on
target dimension (every fixed qubit gate) simply ignores the argument.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..operations import Operation
from ..registers import RegisterRef
from ..resource_layout import DeviceOperand

if TYPE_CHECKING:
    from ..backends.steps import BuiltinKernelKey

DeviceOperands = tuple[DeviceOperand, ...]


class MatrixImplementation:
    """Base class for a matrix-family implementation rule.

    A rule receives the bare :py:class:`~fatqat.operations.Operation` instance that was applied (e.g. an
    `RX(0.3)` value) plus the `targets` :py:class:`~fatqat.registers.RegisterRef` tuple by keyword, and
    returns its local matrix. Most callers never need to subclass this
    directly: `ImplementationMap.add` auto-wraps a plain
    `np.ndarray` (as `FixedMatrix`), a `_DimMatrix`, or a bare
    callable. Subclass and override `__call__` for a stateful or configured
    implementation.
    """

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> np.ndarray:
        raise NotImplementedError

    def _kernel_key(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> "BuiltinKernelKey | None":
        """Return this implementation's canonical kernel identity, or ``None``.

        ``None`` (the default for every rule) means "no declared identity":
        an engine must treat the resolved matrix as opaque content. Only the
        canonical built-in registrations override this, via
        `_KeyedImplementation` - identity is a fact about *which
        implementation was selected*, so it must never be inferred from the
        operation class or the matrix's numeric content.
        """
        return None


class _KeyedImplementation(MatrixImplementation):
    """Delegating wrapper that attaches canonical kernel identity to a rule.

    Applied only by ``default_matrix_implementation_map()`` at registration
    time. The wrapped rule keeps full ownership of matrix production; this
    wrapper adds exactly one fact - the `BuiltinKernelKey` naming which
    canonical implementation the caller selected.
    """

    def __init__(self, rule: MatrixImplementation, key: "BuiltinKernelKey") -> None:
        self._rule = rule
        self._key = key

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...] = ()
    ) -> np.ndarray:
        return self._rule(op, targets=targets)

    def _kernel_key(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> "BuiltinKernelKey | None":
        return self._key


def _validate_square_matrix(matrix: np.ndarray) -> None:
    """Raise `ValueError` unless `matrix` is square with side length >= 2.

    Deliberately does not require a power-of-two side length: `FixedMatrix`
    has no way to know what dimension its caller intends (it never sees the
    target operation or register), and a fixed-dimension restriction here
    would reject legitimate non-qubit matrices (e.g. a qutrit's dim=3 gate)
    with no compensating safety benefit. The arity-aware shape check against
    a specific operation happens separately, in `_wrap_rule`.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"matrix side length must be >= 2, got {n}")


class FixedMatrix(MatrixImplementation):
    """A constant matrix, independent of the applied operation's fields."""

    def __init__(self, matrix: np.ndarray) -> None:
        """Copy, validate, and freeze `matrix` as this rule's constant value.

        Args:
            matrix: Square matrix with side length >= 2. Copied on
                construction, so later mutation of the caller's array does
                not affect this rule.

        Raises:
            ValueError: If `matrix` is not square or its side length is < 2.
        """
        matrix = np.array(matrix, dtype=complex, copy=True)
        _validate_square_matrix(matrix)
        matrix.flags.writeable = False
        self._matrix = matrix

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...] = ()
    ) -> np.ndarray:
        return self._matrix


class _DimMatrix(MatrixImplementation):
    """A rule whose matrix depends only on the target subsystem dimensions.

    Use this when a gate's matrix is fixed *given* the dimensions of its
    targets but is not itself a single constant matrix (e.g. a qudit `Shift`
    gate, whose permutation matrix depends on `targets[0].register.dim`).
    Unlike `FixedMatrix`, this always reads `targets`, so it cannot be used
    with the `targets=()` default; the caller (backend resolution) always
    supplies the real target tuple.
    """

    def __init__(self, fn: "Callable[[tuple[int, ...]], np.ndarray]") -> None:
        """Store `fn`, called on demand with the target dimensions tuple.

        Args:
            fn: Callable taking a `tuple[int, ...]` of target subsystem
                dimensions (in target order) and returning the local matrix.
        """
        self._fn = fn

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> np.ndarray:
        dims = tuple(t.register.dim for t in targets)
        return self._fn(dims)


def _resolve_operation_class(op: Operation | type[Operation]) -> type[Operation]:
    """Normalize an :py:class:`~fatqat.operations.Operation` instance or subclass to its registry key.

    Accepts either an :py:class:`~fatqat.operations.Operation` instance (e.g. `fq.ops.X`) or an :py:class:`~fatqat.operations.Operation`
    subclass (e.g. a custom gate class) and returns the class to key the
    registry by. Applying `type(...)` unconditionally would be wrong for the
    class case: `type(MyGate)` is the metaclass `type`, not `MyGate`.
    """
    if isinstance(op, Operation):
        return type(op)
    if isinstance(op, type) and issubclass(op, Operation):
        return op
    raise TypeError(f"expected an Operation instance or subclass, got {op!r}")


def _require_fixed_arity(op_cls: type[Operation]) -> None:
    """Raise `TypeError` if `op_cls` has variable arity (`_num_subsystems is None`).

    This is a deliberate scope policy, not a technical limit: rules do receive
    `targets` and could in principle size a matrix from `len(targets)`. But a
    variable-arity operation has no single canonical matrix shape to validate
    a rule's output against, so it stays out of scope for this registry unless
    a concrete variadic-matrix gate need appears.
    """
    if op_cls._num_subsystems is None:
        raise TypeError(
            f"{op_cls.__name__} has variable arity (_num_subsystems is None); "
            "the matrix implementation map only supports fixed-arity operations"
        )


def _normalize_device_operands(device_operands: DeviceOperands) -> DeviceOperands:
    """Normalize device operands and verify they can be used as a dict key."""
    key = tuple(device_operands)
    hash(key)
    return key


def _require_device_operands_arity(
    op_cls: type[Operation], device_operands: DeviceOperands
) -> None:
    """Raise `ValueError` if device operands do not match `op_cls` arity.

    Only arity is checked here: the general map does not know what a target
    key element means (an integer device label, a zone name, ...), so it
    cannot range-check or type-check individual elements. That is left to
    the backend that constructs a device-specific map.
    """
    expected = op_cls._num_subsystems
    if len(device_operands) != expected:
        raise ValueError(
            f"{op_cls.__name__} expects {expected} device operand(s), "
            f"got {len(device_operands)}"
        )


def _callable_wants_targets(rule: Callable) -> bool:
    """True if a bare callable declares a `targets` parameter (or **kwargs).

    A rule is targets-aware if it names a `targets` parameter explicitly, or
    accepts arbitrary keyword arguments (`**kwargs`) and so can absorb a
    `targets=` keyword regardless. If the signature cannot be introspected
    (`inspect.signature` can raise `ValueError` or `TypeError` for some
    C-implemented or otherwise uninspectable callables), the callable is
    conservatively treated as not wanting `targets` and called as `rule(op)`.
    """
    try:
        params = inspect.signature(rule).parameters
    except (ValueError, TypeError):
        return False
    if "targets" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class _CallableMatrixImplementation(MatrixImplementation):
    """Adapts a bare `f(op)` or `f(op, targets)` callable to `MatrixImplementation`."""

    def __init__(self, func: Callable, wants_targets: bool) -> None:
        self._func = func
        self._wants_targets = wants_targets

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> np.ndarray:
        if self._wants_targets:
            return self._func(op, targets=targets)
        return self._func(op)


def _wrap_rule(
    op_cls: type[Operation],
    rule: "MatrixImplementation | Callable | np.ndarray",
) -> MatrixImplementation:
    """Normalize an `add()` implementation argument into a `MatrixImplementation`.

    Accepts an already-built `MatrixImplementation` (returned as-is, e.g. a
    `FixedMatrix` or `_DimMatrix`), a plain `np.ndarray` (wrapped in
    `FixedMatrix`, which only requires it be square with side length >= 2,
    see `_validate_square_matrix`), or a bare `f(op)`/`f(op, targets)`
    callable (wrapped). Every stored rule is a `MatrixImplementation`
    instance, so `implementation_for()` always returns a uniform type regardless of how the
    rule was registered.

    A callable is not arity-checked at registration: a rule that cannot be
    called in its detected `f(op)`/`f(op, targets)` shape raises the first
    time it is used, where the backend wraps it in a :py:exc:`~fatqat.errors.MatrixImplementationError`
    naming the operation. Registration only distinguishes the two shapes (via
    `_callable_wants_targets`) so the call site passes `targets=` iff wanted.

    Raises:
        TypeError: If `rule` is none of the above (e.g. a string or a plain
            object), checked explicitly here so the error names the
            operation and the bad value.
    """
    if isinstance(rule, MatrixImplementation):
        return rule
    if isinstance(rule, np.ndarray):
        return FixedMatrix(rule)  # square-only validation lives in FixedMatrix
    if not callable(rule):
        raise TypeError(
            f"rule for {op_cls.__name__} must be a MatrixImplementation, "
            f"np.ndarray, or callable, got {rule!r}"
        )
    return _CallableMatrixImplementation(rule, _callable_wants_targets(rule))


class ImplementationMap:
    """Resolve operation families and device operands to implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map.

        `_unconstrained_rules` holds unconstrained per-operation implementations.
        `_device_operand_rules` holds implementations for explicit device
        operands. An operation family uses at most one mode.
        """
        self._unconstrained_rules: dict[type[Operation], MatrixImplementation] = {}
        self._device_operand_rules: dict[
            type[Operation], dict[DeviceOperands, MatrixImplementation]
        ] = {}

    def add(
        self,
        op: Operation | type[Operation],
        implementation: "MatrixImplementation | Callable | np.ndarray",
        *,
        device_operands: DeviceOperands | None = None,
    ) -> None:
        """Add an unconstrained or device-specific implementation.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance (e.g. `fq.ops.X`) or subclass (e.g. a
                custom gate class). Normalized to the operation's class for
                the registry key.
            implementation: A `MatrixImplementation` instance (e.g. `FixedMatrix` or
                `_DimMatrix`), a bare `np.ndarray` (wrapped in
                `FixedMatrix`), or a bare callable, either `f(op)` or
                `f(op, targets)`, detected by a parameter literally named
                `targets` (or `**kwargs`), returning the operation's matrix
                (wrapped automatically).

        Raises:
            TypeError: If `op` is neither an :py:class:`~fatqat.operations.Operation` instance nor
                subclass, or if its operation class has variable arity. A bare
                callable of the wrong shape is not rejected here; it fails on
                first use (see `_wrap_rule`).
            ValueError: If a bare `np.ndarray` is not square with side
                length >= 2, or if `op` already has device-specific
                registrations, mutually exclusive with `add(..., device_operands=...)`; see
                its docstring for why.
        """
        if device_operands is not None:
            self._add_for_device_operands(op, device_operands, implementation)
            return

        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._device_operand_rules:
            raise ValueError(
                f"{op_cls.__name__} already has device-specific implementations, "
                "cannot also add an unconstrained implementation for "
                "the same operation. Call remove(op) first if you want "
                "to replace its registrations."
            )
        self._unconstrained_rules[op_cls] = _wrap_rule(op_cls, implementation)

    def _add_for_device_operands(
        self,
        op: Operation | type[Operation],
        device_operands: DeviceOperands,
        implementation: "MatrixImplementation | Callable | np.ndarray",
    ) -> None:
        """Add an implementation for one operation and device-operand tuple.

        `add` supports two mutually exclusive modes per operation family:
        one unconstrained implementation, or explicit implementations for
        device-operand tuples. An absent tuple in the latter mode is illegal;
        call `remove(op)` before switching modes.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance or subclass. Normalized to the
                operation's class for the registry key, same as `add`.
            device_operands: An ordered hashable tuple identifying the device-level
                target (e.g. a flat integer subsystem tuple like `(0, 1)`).
                Its length must match the operation's arity; its element
                types and values are not otherwise validated here; that is
                a device-specific concern owned by the caller.
            implementation: Same accepted shapes as `add`.

        Raises:
            TypeError: If `op` is neither an :py:class:`~fatqat.operations.Operation` instance nor
                subclass, or if its operation class has variable arity.
            ValueError: If `device_operands`' length does not match the
                operation's arity, if a bare `np.ndarray` rule is not square
                with side length >= 2, or if `op` already has an unconstrained
                rule (see above).
        """
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._unconstrained_rules:
            raise ValueError(
                f"{op_cls.__name__} already has an unconstrained rule "
                "(add); cannot also add a device-specific implementation for "
                "the same operation. Call remove(op) first if you want "
                "to replace its registrations."
            )
        operands = _normalize_device_operands(device_operands)
        _require_device_operands_arity(op_cls, operands)
        self._device_operand_rules.setdefault(op_cls, {})[operands] = _wrap_rule(
            op_cls, implementation
        )

    def supports(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> bool:
        """Return whether this map has any rule for the operation family.

        True if the operation has an unconstrained implementation (`add`) or a
        device-specific implementation for at least one operand tuple.
        The two are mutually exclusive per operation, so never both. Does
        not check whether any particular device operands is legal; use `implementation_for`
        for that, or `device_operands_for` to distinguish uniform from explicit
        support.
        """
        if device_operands is not None:
            return (
                self.implementation_for(op, device_operands=device_operands) is not None
            )
        op_cls = _resolve_operation_class(op)
        return (
            op_cls in self._unconstrained_rules or op_cls in self._device_operand_rules
        )

    def implementation_for(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> MatrixImplementation | None:
        """Return the matrix implementation selected for an operation.

        Always a `MatrixImplementation` instance regardless of what was
        registered: a bare callable is wrapped, a bare ndarray becomes a
        `FixedMatrix`.

        With `device_operands` omitted, only the unconstrained `add()` implementation is
        consulted, regardless of any device-specific implementations for the
        operation. With `device_operands` given: if the operation has any
        device-specific implementations, only those are consulted. `None` means
        the operation family is supported but this specific device operands is
        not legal. If the operation has no device-specific implementations at
        all, the unconstrained `add()` implementation (if any) is returned for
        every device operands. This is what keeps maps with only an unconstrained implementation working
        unchanged under device-specific lookup.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance or subclass.
            device_operands: An ordered hashable tuple identifying the device-level
                target. Omit to look up only the unconstrained rule.
        """
        op_cls = _resolve_operation_class(op)
        if device_operands is None:
            return self._unconstrained_rules.get(op_cls)
        table = self._device_operand_rules.get(op_cls)
        if table is not None:
            return table.get(_normalize_device_operands(device_operands))
        return self._unconstrained_rules.get(op_cls)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return every operation family with at least one implementation."""
        return frozenset(self._unconstrained_rules | self._device_operand_rules)

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        """Return the finite set of device operands selected for an operation.

        Empty if the operation has no device-specific implementations, including
        when it has an unconstrained `add()` implementation instead, which has no
        fixed set of legal keys. Combine with `supports` to tell the two
        apart: `supports(op) and not device_operands_for(op)` means `op` is legal
        on any target of the correct arity (uniform); a non-empty result
        means legal only on those keys; `not supports(op)` means not
        supported at all.
        """
        op_cls = _resolve_operation_class(op)
        return frozenset(self._device_operand_rules.get(op_cls, ()))

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove a registered matrix implementation, if present.

        Removes both the unconstrained rule and any device-specific implementations for
        this operation.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance or subclass to remove. Removing an
                operation that was never registered is a no-op.
        """
        op_cls = _resolve_operation_class(op)
        self._unconstrained_rules.pop(op_cls, None)
        self._device_operand_rules.pop(op_cls, None)

    def copy(self) -> "ImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects themselves are shared (not deep-copied) between the
        original and the copy; rules are expected to be immutable or
        self-contained, so sharing them across independent map copies is
        safe. Mutating one map's registrations (`add`/`add(..., device_operands=...)`/
        `remove`) never affects the other. The per-operation device-operand
        tables are copied individually (not just the outer dict), so mutating
        one map's device-specific implementations for an operation cannot leak
        into the other map's table for that same operation.
        """
        clone = ImplementationMap()
        clone._unconstrained_rules = dict(self._unconstrained_rules)
        clone._device_operand_rules = {
            op_cls: dict(operand_rules)
            for op_cls, operand_rules in self._device_operand_rules.items()
        }
        return clone
