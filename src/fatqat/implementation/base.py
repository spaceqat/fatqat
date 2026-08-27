"""Matrix-implementation abstraction: the rule protocol, its wrappers, and the
implementation map (unconstrained and device-specific).

A matrix implementation maps an operation to its local matrix (physics only).
The backend pairs that matrix with layout-resolved target indices to build an
``ApplyMatrixStep`` (see ``_backends.steps``), the plain data container the
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
from ._operation_registry import (
    DeviceOperands,
    _OperationRuleRegistry,
)

if TYPE_CHECKING:
    from .._backends.steps import BuiltinKernelKey


class MatrixImplementation:
    """Base class for a matrix-family implementation rule.

    A rule receives the bare :py:class:`~fatqat.operations.Operation` instance that was applied (e.g. an
    `RX(0.3)` value) plus the `targets` :py:class:`~fatqat.registers.RegisterRef` tuple by keyword, and
    returns its local matrix. Most callers never need to subclass this
    directly: `MatrixImplementationMap.add` auto-wraps a plain
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


class MatrixImplementationMap:
    """Resolve operation families and device operands to implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map.

        Composes `_OperationRuleRegistry` for its unconstrained-versus-device-
        specific storage mechanics, shared with every other implementation-map
        family; this class owns only matrix-specific rule wrapping, public
        documentation, and error wording.
        """
        self._registry: _OperationRuleRegistry[MatrixImplementation] = (
            _OperationRuleRegistry()
        )

    def add(
        self,
        op: Operation | type[Operation],
        implementation: "MatrixImplementation | Callable | np.ndarray",
        *,
        device_operands: DeviceOperands | None = None,
    ) -> None:
        """Add an unconstrained or device-specific implementation.

        Matrix rows and columns use the ordered targets as their local factor
        order: targets[0] is most significant and targets[-1] is least
        significant. For local basis digits (b0, ..., bk) with dimensions
        (d0, ..., dk), the flat index is b0 * (d1 * ... * dk) +
        b1 * (d2 * ... * dk) + ... + bk, so the last target changes fastest.
        This local convention is independent of the simulator's full-system
        little-endian basis order.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance (e.g. `ops.X`) or subclass (e.g. a
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
            self._registry.add_device_operands(
                op,
                device_operands,
                lambda op_cls: _wrap_rule(op_cls, implementation),
            )
            return
        self._registry.add_unconstrained(
            op, lambda op_cls: _wrap_rule(op_cls, implementation)
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
        return self._registry.supports(op)

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
        return self._registry.get(op, device_operands=device_operands)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return every operation family with at least one implementation."""
        return self._registry.supported_operations()

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
        return self._registry.device_operands_for(op)

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove a registered matrix implementation, if present.

        Removes both the unconstrained rule and any device-specific implementations for
        this operation.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance or subclass to remove. Removing an
                operation that was never registered is a no-op.
        """
        self._registry.remove(op)

    def copy(self) -> "MatrixImplementationMap":
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
        clone = MatrixImplementationMap()
        clone._registry = self._registry.copy()
        return clone
