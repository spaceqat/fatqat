"""Matrix, channel, and custom-operation rules with device-aware lookup.

A matrix rule receives an applied operation and its ordered program targets, then
returns the operation's local matrix. A map can register one rule for every
device target or separate rules for specific ordered device labels.

The first program target is the most-significant local matrix factor and the
last target changes fastest. Controlled operations follow FATQAT's usual
control-first target order.
"""

from __future__ import annotations

import inspect
from math import prod
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..errors import BackendValidationError, MatrixImplementationError
from ..operations import CustomOperation, Operation
from .custom import CustomOperationImplementation
from ..registers import RegisterRef
from .base_map import (
    DeviceOperands,
    _BaseImplementationMap,
)

if TYPE_CHECKING:
    from .._backends.steps import BuiltinKernelKey
    from ..noise.base import Channel


def _resolve_implementation_class(
    descriptor: Operation | Channel | type[Operation] | type[Channel],
) -> type[Operation] | type[Channel]:
    """Normalize a simulator implementation key without coupling pulse maps to noise."""
    # Noise descriptors also use implementation.matrices; import after module setup.
    from ..noise.base import Channel

    cls = descriptor if isinstance(descriptor, type) else type(descriptor)
    if not issubclass(cls, (Operation, Channel)):
        raise TypeError(
            f"expected an Operation or Channel instance or subclass, got {descriptor!r}"
        )
    return cls


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

    def _batch_builder(self, op: Operation) -> "_BatchedBuilder | None":
        """Return a vectorized builder for sweeps of ``op``, or ``None``.

        ``None`` (the default for every rule) means sweep replay calls the
        rule once per row. Only canonical built-in registrations override
        this, via `_KeyedImplementation`.
        """
        return None


# A batched builder receives a field accessor (a float64 column for a
# `Parameter` field, the declared value otherwise) and the program targets,
# and returns an ``(rows, d, d)`` matrix stack.
_BatchedBuilder = Callable[[Callable[[str], Any], tuple[RegisterRef, ...]], np.ndarray]


class _KeyedImplementation(MatrixImplementation):
    """Delegating wrapper that attaches canonical kernel identity to a rule.

    Applied only by ``default_matrix_implementation_map()`` at registration
    time. The wrapped rule keeps full ownership of matrix production; this
    wrapper adds exactly one fact - the `BuiltinKernelKey` naming which
    canonical implementation the caller selected - plus, for a parametric
    gate, the batched twin of its rule used by sweep replay.

    The batched twin applies only to exactly ``op_cls``. A keyed rule can be
    re-registered for another class, which may override `validate_targets`
    or the fields the builder reads. The built-in classes validate targets
    without reading their angles, and a template is validated when it joins
    a program, so batched replay has nothing to re-check.
    """

    def __init__(
        self,
        rule: MatrixImplementation,
        key: "BuiltinKernelKey",
        *,
        op_cls: type[Operation] | None = None,
        batch: _BatchedBuilder | None = None,
    ) -> None:
        self._rule = rule
        self._key = key
        self._op_cls = op_cls
        self._batch = batch

    def __call__(
        self, op: Operation, *, targets: tuple[RegisterRef, ...] = ()
    ) -> np.ndarray:
        return self._rule(op, targets=targets)

    def _kernel_key(
        self, op: Operation, *, targets: tuple[RegisterRef, ...]
    ) -> "BuiltinKernelKey | None":
        return self._key

    def _batch_builder(self, op: Operation) -> _BatchedBuilder | None:
        if self._batch is None or type(op) is not self._op_cls:
            return None
        return self._batch


def _validate_matrix_shape(
    shape: tuple[int, ...], op: Operation, target_dims: tuple[int, ...]
) -> None:
    """Check a resolved local matrix shape against its target dimensions."""
    expected = prod(target_dims)
    if shape != (expected, expected):
        raise BackendValidationError(
            f"{type(op).__name__} resolved to a {shape} matrix, incompatible "
            f"with target dimensions {target_dims} (expected "
            f"{(expected, expected)})"
        )


def _build_matrix(
    rule: MatrixImplementation,
    op: Operation,
    *,
    targets: tuple[RegisterRef, ...],
    target_dims: tuple[int, ...],
) -> np.ndarray:
    """Call a matrix rule for one applied operation and validate its shape.

    Rule failures raise `MatrixImplementationError` naming the operation; a
    wrongly shaped matrix raises `BackendValidationError`.
    """
    try:
        matrix = rule(op, targets=targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(op).__name__} raised: {exc}"
        ) from exc
    _validate_matrix_shape(matrix.shape, op, target_dims)
    return matrix


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
    op_cls: type[Operation] | type[Channel],
    rule: "MatrixImplementation | CustomOperationImplementation | Callable | np.ndarray",
) -> MatrixImplementation | CustomOperationImplementation | Callable:
    """Normalize matrix rules and retain channel or custom-operation rules.

    Accepts an already-built `MatrixImplementation` (returned as-is, e.g. a
    `FixedMatrix` or `_DimMatrix`), a plain `np.ndarray` (wrapped in
    `FixedMatrix`, which only requires it be square with side length >= 2,
    see `_validate_square_matrix`), or a bare `f(op)`/`f(op, targets)`
    callable (wrapped). These become MatrixImplementation instances. CustomOperation
    classes instead require CustomOperationImplementation, retained unchanged.
    Channel classes require a callable, also retained unchanged.

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
    if not issubclass(op_cls, Operation):
        if not callable(rule):
            raise TypeError(
                f"rule for {op_cls.__name__} must be callable, got {rule!r}"
            )
        return rule
    if issubclass(op_cls, CustomOperation):
        if not isinstance(rule, CustomOperationImplementation):
            raise TypeError("CustomOperation requires a CustomOperationImplementation")
        return rule
    if isinstance(rule, CustomOperationImplementation):
        raise TypeError("custom implementations require a CustomOperation class")
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


class MatrixImplementationMap(
    _BaseImplementationMap[
        MatrixImplementation | CustomOperationImplementation | Callable
    ]
):
    """Map operation and channel classes to simulator implementations.

    Registration and lookup accept operation or channel instances and classes.
    A new map is empty. Register either one uniform rule or one or more
    device-specific rules for an operation family; the two registration modes
    cannot be mixed for the same operation. ``CustomOperation`` classes require
    ``CustomOperationImplementation`` objects. Channel rules return Kraus matrices.
    """

    def __init__(self) -> None:
        """Create an empty implementation map."""
        super().__init__(_resolve_implementation_class)

    def add(
        self,
        op: Operation | Channel | type[Operation] | type[Channel],
        implementation: "MatrixImplementation | CustomOperationImplementation | Callable | np.ndarray",
        *,
        device_operands: DeviceOperands | None = None,
    ) -> None:
        """Add a uniform or device-specific operation or channel implementation.

        Matrix rows and columns follow program target order: `targets[0]` is
        the most-significant local factor and the last target changes fastest.
        For controlled operations, controls come before targets.

        Args:
            op: Operation or Channel instance or subclass.
                Instances and their classes select the same operation family.
            implementation: The rule for the registered operation or channel:

                - Matrix operation: a `MatrixImplementation`, a NumPy array,
                  or a callable. Arrays become `FixedMatrix` rules. Callables
                  receive `rule(op)`, or `rule(op, targets=targets)` if they
                  accept a `targets=` keyword or `**kwargs`.
                - Channel: a callable receiving `rule(channel, targets=targets)`
                  and returning a nonempty tuple of Kraus matrices.
                - Custom operation: a `CustomOperationImplementation` object
                  defining `execute(operation, state, context)`.
            device_operands: Ordered, hashable device labels for one physical
                target tuple. Its length must match a fixed integer
                ``num_subsystems``. Instance-dependent and variable widths match
                the exact registered tuple when applied. Omit for a uniform rule.

        Raises:
            TypeError: If `op` is not an operation or channel instance or subclass,
                the operation is a direct-control operation,
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
            self._add_device_operands(
                op,
                device_operands,
                lambda op_cls: _wrap_rule(op_cls, implementation),
            )
            return
        self._add_unconstrained(
            op,
            lambda op_cls: _wrap_rule(op_cls, implementation),
        )

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return operation classes that have at least one registered rule."""
        return frozenset(
            cls for cls in self._registered_types() if issubclass(cls, Operation)
        )

    def supported_channels(self) -> frozenset[type[Channel]]:
        """Return channel classes that have at least one registered rule."""
        return frozenset(
            cls for cls in self._registered_types() if not issubclass(cls, Operation)
        )
