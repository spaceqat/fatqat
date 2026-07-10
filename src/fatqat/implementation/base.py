"""Matrix-implementation abstraction: the rule protocol, its wrappers, and the
class-keyed registry.

A matrix implementation maps an operation to its local matrix (physics only).
The backend pairs that matrix with layout-resolved target indices to build an
``ApplyMatrixStep`` (see ``backends.steps``) — the plain data container the
statevector engine reads directly.

Local matrix convention (binding for every entry in this package):
    - ``AppliedOperation.targets`` operand order defines the local
      tensor-factor order; ``targets[0]`` is the local most-significant bit
      (MSB), ``targets[-1]`` the local least-significant bit (LSB). See
      ``engine._apply_matrix`` for the little-endian contraction this feeds.
    - For every controlled gate (``CX``, ``CZ``, ``CY``, and the controlled
      gates added in later batches), the control operand(s) come first and the
      target operand(s) come last — operand 0 (and operand 1 for
      doubly-controlled gates) is the control, occupying the local MSB
      position(s).

A matrix implementation rule receives the bare `Operation` instance that was
applied (e.g. `RX(0.3)`) plus the `targets: tuple[RegisterRef, ...]` operand
tuple by keyword, and returns the local matrix — never the surrounding
`AppliedOperation`, and never a feedforward `condition`: condition resolution
happens separately, in the backend. `targets` lets a rule read
`targets[0].register.dim` to build a dimension-dependent matrix (e.g. a
qudit `Shift`/`Clock`/`Sum` gate); a rule whose matrix never depends on
target dimension (every fixed qubit gate) simply ignores the argument.
"""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from typing import Callable

import numpy as np

from ..operations import Operation
from ..registers import RegisterRef

TargetKey = tuple[Hashable, ...]


class MatrixImplementation:
    """Base class for a matrix-family implementation rule.

    A rule receives the bare `Operation` instance that was applied (e.g. an
    `RX(0.3)` value) plus the `targets` `RegisterRef` tuple by keyword, and
    returns its local matrix. Most callers never need to subclass this
    directly: `MatrixImplementationMap.register` auto-wraps a plain
    `np.ndarray` (as `FixedMatrix`), a `_DimMatrix`, or a bare
    callable. Subclass and override `__call__` for a stateful or configured
    implementation.
    """

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
        raise NotImplementedError


def _validate_square_matrix(matrix: np.ndarray) -> None:
    """Raise `ValueError` unless `matrix` is square with side length >= 2.

    Deliberately does not require a power-of-two side length: `FixedMatrix`
    has no way to know what dimension its caller intends (it never sees the
    target operation or register), and a fixed-dimension restriction here
    would reject legitimate non-qubit matrices (e.g. a qutrit's dim=3 gate)
    with no compensating safety benefit — the arity-aware shape check against
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

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...] = ()) -> np.ndarray:
        return self._matrix


class _DimMatrix(MatrixImplementation):
    """A rule whose matrix depends only on the target subsystem dimensions.

    Use this when a gate's matrix is fixed *given* the dimensions of its
    targets but is not itself a single constant matrix (e.g. a qudit `Shift`
    gate, whose permutation matrix depends on `targets[0].register.dim`).
    Unlike `FixedMatrix`, this always reads `targets`, so it cannot be used
    with the `targets=()` default — the caller (backend resolution) always
    supplies the real target tuple.
    """

    def __init__(self, fn: "Callable[[tuple[int, ...]], np.ndarray]") -> None:
        """Store `fn`, called on demand with the target dimensions tuple.

        Args:
            fn: Callable taking a `tuple[int, ...]` of target subsystem
                dimensions (in target order) and returning the local matrix.
        """
        self._fn = fn

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
        dims = tuple(t.register.dim for t in targets)
        return self._fn(dims)


def _resolve_operation_class(op: Operation | type[Operation]) -> type[Operation]:
    """Normalize an `Operation` instance or subclass to its registry key.

    Accepts either an `Operation` instance (e.g. `fq.ops.X`) or an `Operation`
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


def _normalize_target_key(target_key: TargetKey) -> TargetKey:
    """Normalize a device target key and verify it can be used as a dict key."""
    key = tuple(target_key)
    hash(key)
    return key


def _require_target_key_arity(op_cls: type[Operation], target_key: TargetKey) -> None:
    """Raise `ValueError` if `target_key`'s length does not match `op_cls` arity.

    Only arity is checked here: the general map does not know what a target
    key element means (an integer device label, a zone name, ...), so it
    cannot range-check or type-check individual elements. That is left to
    the backend that constructs a device-specific map.
    """
    expected = op_cls._num_subsystems
    if len(target_key) != expected:
        raise ValueError(
            f"{op_cls.__name__} expects {expected} target key element(s), "
            f"got {len(target_key)}"
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

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
        if self._wants_targets:
            return self._func(op, targets=targets)
        return self._func(op)


def _wrap_rule(
    op_cls: type[Operation],
    rule: "MatrixImplementation | Callable | np.ndarray",
) -> MatrixImplementation:
    """Normalize a `register()` rule argument into a `MatrixImplementation`.

    Accepts an already-built `MatrixImplementation` (returned as-is, e.g. a
    `FixedMatrix` or `_DimMatrix`), a plain `np.ndarray` (wrapped in
    `FixedMatrix`, which only requires it be square with side length >= 2 —
    see `_validate_square_matrix`), or a bare `f(op)`/`f(op, targets)`
    callable (wrapped). Every stored rule is a `MatrixImplementation`
    instance, so `get()` always returns a uniform type regardless of how the
    rule was registered.

    A callable is not arity-checked at registration: a rule that cannot be
    called in its detected `f(op)`/`f(op, targets)` shape raises the first
    time it is used, where the backend wraps it in a `MatrixImplementationError`
    naming the operation. Registration only distinguishes the two shapes (via
    `_callable_wants_targets`) so the call site passes `targets=` iff wanted.

    Raises:
        TypeError: If `rule` is none of the above (e.g. a string or a plain
            object) — checked explicitly here so the error names the
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


class MatrixImplementationMap:
    """Class-keyed registry from operation classes to matrix implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map.

        `_rules` holds uniform per-operation rules (`register`);
        `_target_rules` holds per-target-key rules (`register_for`). An
        operation lives in at most one of the two — see `register` and
        `register_for` for the mutual-exclusion rule and `get` for how each
        mode resolves.
        """
        self._rules: dict[type[Operation], MatrixImplementation] = {}
        self._target_rules: dict[type[Operation], dict[TargetKey, MatrixImplementation]] = {}

    def register(
        self,
        op: Operation | type[Operation],
        rule: "MatrixImplementation | Callable | np.ndarray",
    ) -> None:
        """Register a matrix implementation for an operation.

        Args:
            op: An `Operation` instance (e.g. `fq.ops.X`) or subclass (e.g. a
                custom gate class). Normalized to the operation's class for
                the registry key.
            rule: A `MatrixImplementation` instance (e.g. `FixedMatrix` or
                `_DimMatrix`), a bare `np.ndarray` (wrapped in
                `FixedMatrix`), or a bare callable — either `f(op)` or
                `f(op, targets)`, detected by a parameter literally named
                `targets` (or `**kwargs`) — returning the operation's matrix
                (wrapped automatically).

        Raises:
            TypeError: If `op` is neither an `Operation` instance nor
                subclass, or if its operation class has variable arity. A bare
                callable of the wrong shape is not rejected here; it fails on
                first use (see `_wrap_rule`).
            ValueError: If a bare `np.ndarray` is not square with side
                length >= 2, or if `op` already has target-aware
                registrations — mutually exclusive with `register_for`, see
                its docstring for why.
        """
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._target_rules:
            raise ValueError(
                f"{op_cls.__name__} already has target-aware registrations "
                "(register_for); cannot also register a class-keyed rule for "
                "the same operation. Call unregister(op) first if you want "
                "to replace its registrations."
            )
        self._rules[op_cls] = _wrap_rule(op_cls, rule)

    def register_for(
        self,
        op: Operation | type[Operation],
        target_key: TargetKey,
        rule: "MatrixImplementation | Callable | np.ndarray",
    ) -> None:
        """Register a matrix implementation for one operation on one device target key.

        Once an operation has any target-aware registration, `get` stops
        falling back to a class-keyed `register()` rule for that
        operation: an absent target key means the target is illegal, not
        that the caller should fall back to a default. `register` and
        `register_for` are therefore mutually exclusive per operation, each
        raising if the other already has an entry — `unregister(op)` first
        if you need to switch modes. Calling `register_for` again for an
        operation that already has target-aware entries is fine and normal
        (e.g. one call per grid edge).

        Args:
            op: An `Operation` instance or subclass. Normalized to the
                operation's class for the registry key, same as `register`.
            target_key: A hashable tuple identifying the device-level
                target (e.g. a flat integer subsystem tuple like `(0, 1)`).
                Its length must match the operation's arity; its element
                types and values are not otherwise validated here — that is
                a device-specific concern owned by the caller.
            rule: Same accepted shapes as `register`.

        Raises:
            TypeError: If `op` is neither an `Operation` instance nor
                subclass, or if its operation class has variable arity.
            ValueError: If `target_key`'s length does not match the
                operation's arity, if a bare `np.ndarray` rule is not square
                with side length >= 2, or if `op` already has a class-keyed
                rule (see above).
        """
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._rules:
            raise ValueError(
                f"{op_cls.__name__} already has a class-keyed rule "
                "(register); cannot also register a target-aware rule for "
                "the same operation. Call unregister(op) first if you want "
                "to replace its registrations."
            )
        key = _normalize_target_key(target_key)
        _require_target_key_arity(op_cls, key)
        self._target_rules.setdefault(op_cls, {})[key] = _wrap_rule(op_cls, rule)

    def supports(self, op: Operation | type[Operation]) -> bool:
        """Return whether this map has any rule for the operation family.

        True if the operation has a class-keyed rule (`register`) or a
        target-aware rule for at least one target key (`register_for`) —
        the two are mutually exclusive per operation, so never both. Does
        not check whether any particular target key is legal — use `get`
        for that.
        """
        op_cls = _resolve_operation_class(op)
        return op_cls in self._rules or op_cls in self._target_rules

    def get(
        self,
        op: Operation | type[Operation],
        target_key: TargetKey | None = None,
    ) -> MatrixImplementation | None:
        """Return the matrix implementation registered for an operation.

        Always a `MatrixImplementation` instance regardless of what was
        registered — a bare callable is wrapped, a bare ndarray becomes a
        `FixedMatrix`.

        With `target_key` omitted, only the class-keyed `register()` rule is
        consulted, regardless of any target-aware registrations for the
        operation. With `target_key` given: if the operation has any
        target-aware registrations, only those are consulted — `None` means
        the operation family is supported but this specific target key is
        not legal. If the operation has no target-aware registrations at
        all, the class-keyed `register()` rule (if any) is returned for
        every target key — this is what keeps `register`-only maps working
        unchanged under target-aware lookup.

        Args:
            op: An `Operation` instance or subclass.
            target_key: A hashable tuple identifying the device-level
                target. Omit to look up only the class-keyed rule.
        """
        op_cls = _resolve_operation_class(op)
        if target_key is None:
            return self._rules.get(op_cls)
        table = self._target_rules.get(op_cls)
        if table is not None:
            return table.get(_normalize_target_key(target_key))
        return self._rules.get(op_cls)

    def target_keys(self, op: Operation | type[Operation]) -> frozenset[TargetKey]:
        """Return the finite set of target keys registered for an operation.

        Empty if the operation has no target-aware registrations, even if it
        has a class-keyed `register()` rule (that rule has no fixed set of
        legal target keys — see `get`).
        """
        op_cls = _resolve_operation_class(op)
        return frozenset(self._target_rules.get(op_cls, ()))

    def unregister(self, op: Operation | type[Operation]) -> None:
        """Remove a registered matrix implementation, if present.

        Removes both the class-keyed rule and any target-aware rules for
        this operation.

        Args:
            op: An `Operation` instance or subclass to remove. Removing an
                operation that was never registered is a no-op.
        """
        op_cls = _resolve_operation_class(op)
        self._rules.pop(op_cls, None)
        self._target_rules.pop(op_cls, None)

    def copy(self) -> "MatrixImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects themselves are shared (not deep-copied) between the
        original and the copy — rules are expected to be immutable or
        self-contained, so sharing them across independent map copies is
        safe. Mutating one map's registrations (`register`/`register_for`/
        `unregister`) never affects the other. The per-operation target-key
        tables are copied individually (not just the outer dict), so mutating
        one map's target-aware registrations for an operation cannot leak
        into the other map's table for that same operation.
        """
        clone = MatrixImplementationMap()
        clone._rules = dict(self._rules)
        clone._target_rules = {
            op_cls: dict(target_rules) for op_cls, target_rules in self._target_rules.items()
        }
        return clone
