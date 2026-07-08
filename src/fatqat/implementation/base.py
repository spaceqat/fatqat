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
from typing import Callable

import numpy as np

from ..operations import Operation
from ..registers import RegisterRef


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
        """Create an empty implementation map."""
        self._rules: dict[type[Operation], MatrixImplementation] = {}

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
                length >= 2.
        """
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        self._rules[op_cls] = _wrap_rule(op_cls, rule)

    def unregister(self, op: Operation | type[Operation]) -> None:
        """Remove a registered matrix implementation, if present.

        Args:
            op: An `Operation` instance or subclass to remove. Removing an
                operation that was never registered is a no-op.
        """
        op_cls = _resolve_operation_class(op)
        self._rules.pop(op_cls, None)

    def get(self, op: Operation | type[Operation]) -> MatrixImplementation | None:
        """Return the matrix implementation registered for an operation, if any.

        Always a `MatrixImplementation` instance regardless of what was
        registered — a bare callable is wrapped, a bare ndarray becomes a
        `FixedMatrix`.

        Args:
            op: An `Operation` instance or subclass.
        """
        op_cls = _resolve_operation_class(op)
        return self._rules.get(op_cls)

    def copy(self) -> "MatrixImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects themselves are shared (not deep-copied) between the
        original and the copy — rules are expected to be immutable or
        self-contained, so sharing them across independent map copies is
        safe. Mutating one map's registrations (`register`/`unregister`)
        never affects the other.
        """
        clone = MatrixImplementationMap()
        clone._rules = dict(self._rules)
        return clone
