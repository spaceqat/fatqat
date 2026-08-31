"""Matrix rules and device-aware rule maps.

A rule receives an applied operation and its ordered program targets, then
returns the operation's local matrix. A map can register one rule for every
device target or separate rules for specific ordered device labels.

The first program target is the most-significant local matrix factor and the
last target changes fastest. Controlled operations follow FATQAT's usual
control-first target order.
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
    """Base class for reusable matrix rules.

    Override `__call__` to return the local matrix for an applied operation.
    The operation value carries parameters such as a rotation angle. The
    keyword-only `targets` tuple lets a rule inspect each target register's
    dimension. For a stateless rule, pass a plain callable directly to
    `MatrixImplementationMap.add` instead.
    """

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> np.ndarray:
        """Return the local matrix for one applied operation.

        Args:
            op: Applied operation value.
            targets: Scalar program targets in operand order.

        Returns:
            A square complex matrix with side length equal to the product of
            the target dimensions.
        """
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
    a specific operation happens separately, when a backend first resolves
    the rule for a concrete target (`_wrap_rule` itself does not check
    shapes).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"matrix side length must be >= 2, got {n}")


class FixedMatrix(MatrixImplementation):
    """A copied, read-only matrix used for every matching operation.

    `matrix` must be square with side length at least two. It need not have a
    power-of-two size, so fixed qutrit and other qudit matrices are accepted.
    FATQAT copies the input as a complex array and makes that copy read-only;
    later changes to the input array do not affect the rule.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Create a constant rule from `matrix`.

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
        """Return the rule's read-only matrix."""
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
    """Map fixed-arity operation families to local matrix rules.

    A new map is empty. Register either one uniform rule or one or more
    device-specific rules for an operation family; the two registration modes
    cannot be mixed for the same operation.
    """

    def __init__(self) -> None:
        """Create an empty implementation map."""
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
        """Add a uniform or device-specific matrix rule.

        Matrix rows and columns follow program target order: `targets[0]` is
        the most-significant local factor and the last target changes fastest.
        For controlled operations, controls come before targets.

        Args:
            op: Operation instance, such as `ops.X`, or a fixed-arity
                `Operation` subclass. Instances and their classes select the
                same operation family.
            implementation: A `MatrixImplementation`, a NumPy array, or a
                callable. An array becomes a `FixedMatrix`. A callable is
                called as `rule(op)` unless it accepts a `targets=` keyword or
                `**kwargs`; in that case FATQAT calls it as
                `rule(op, targets=targets)`.
            device_operands: Ordered, hashable device labels for one physical
                target tuple. Its length must equal the operation arity. Omit
                it to make the rule apply uniformly.

        Raises:
            TypeError: If `op` is not an operation instance or subclass, the
                operation has variable arity or is a direct-control operation,
                `implementation` has an unsupported form, or a device label is
                not hashable.
            ValueError: If an array is not square with side length at least
                two, the device-operand count does not match the operation
                arity, or this registration would mix uniform and
                device-specific rules for one operation family.

        A callable's invocation shape and returned matrix size are checked
        only when a backend first uses the rule.
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
        """Return whether the operation has a matching rule.

        With `device_operands` omitted, this reports whether the operation
        family has any uniform or device-specific registration. With device
        operands supplied, it reports whether that exact tuple is supported;
        a uniform rule supports every tuple.

        Args:
            op: Operation instance or subclass.
            device_operands: Optional ordered device-label tuple.
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

        The return value is always a `MatrixImplementation`, including when an
        array or callable was originally registered.

        Without `device_operands`, only a uniform rule is returned; a family
        that has only device-specific rules returns `None`. With a tuple,
        FATQAT returns its exact device-specific rule when that registration
        mode is in use, or falls back to the family's uniform rule. `None`
        means no rule matches the requested lookup.

        Args:
            op: Operation instance or subclass.
            device_operands: Ordered device labels. Omit to look up only the
                uniform rule.

        Returns:
            The selected rule, or `None` when none matches.
        """
        return self._registry.get(op, device_operands=device_operands)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return operation classes that have at least one registered rule."""
        return self._registry.supported_operations()

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        """Return every explicitly registered device-label tuple for `op`.

        The result is empty both for an unsupported operation and for an
        operation with a uniform rule. Use `supports(op)` to distinguish
        those cases.
        """
        return self._registry.device_operands_for(op)

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove all rules for an operation family, if present.

        This removes either the uniform rule or every device-specific rule.
        Removing an unregistered operation is a no-op.

        Args:
            op: Operation instance or subclass.
        """
        self._registry.remove(op)

    def copy(self) -> "MatrixImplementationMap":
        """Return a map whose registrations can be edited independently.

        Adding or removing rules on either map does not affect the other.
        Registered rule objects are shared rather than deep-copied, so a
        stateful custom rule remains the same object in both maps.
        """
        clone = MatrixImplementationMap()
        clone._registry = self._registry.copy()
        return clone
