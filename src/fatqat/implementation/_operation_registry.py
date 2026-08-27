"""Private operation-key/device-operand registry mechanics shared by every
implementation-map family (matrix, pulse, ...).

Kept private and generic over each family's own rule type: `MatrixImplementationMap`
(`base.py`) composes `_OperationRuleRegistry` for `MatrixImplementation`
rules; the pulse implementation map composes it for pulse rules. Neither
family's rule wrapping, public documentation, or error wording lives here -
only the mechanics that are identical regardless of what a rule returns:
operation instance/class normalization, fixed-arity checking, ordered
device-operand normalization, the mutually exclusive unconstrained-versus-
device-specific storage policy, lookup, enumeration, removal, and independent
copying.
"""

from __future__ import annotations

from typing import Callable, Generic, Protocol, TypeVar

from ..errors import UnsupportedOperationError
from ..operations import Operation
from ..resource_layout import DeviceOperand

type DeviceOperands = tuple[DeviceOperand, ...]

T = TypeVar("T")


def _resolve_operation_class(op: Operation | type[Operation]) -> type[Operation]:
    """Normalize an :py:class:`~fatqat.operations.Operation` instance or subclass to its registry key.

    Accepts either an :py:class:`~fatqat.operations.Operation` instance (e.g. `ops.X`) or an :py:class:`~fatqat.operations.Operation`
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
    """Raise `TypeError` if `op_cls` has variable arity (`num_subsystems is None`).

    This is a deliberate scope policy, not a technical limit: rules do receive
    `targets` and could in principle size their result from `len(targets)`.
    But a variable-arity operation has no single canonical arity to validate a
    device-operand key or a rule's output shape against, so it stays out of
    scope for every implementation-map family unless a concrete variadic need
    appears.
    """
    if op_cls.num_subsystems is None:
        raise TypeError(
            f"{op_cls.__name__} has variable arity (num_subsystems is None); "
            "implementation maps only support fixed-arity operations"
        )
    if op_cls._is_direct_control:
        raise TypeError(
            f"{op_cls.__name__} is a direct-control operation and cannot have "
            "an implementation-map registration"
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

    Only arity is checked here: the general registry does not know what a
    target key element means (an integer device label, a zone name, ...), so
    it cannot range-check or type-check individual elements. That is left to
    the backend that constructs a device-specific map.
    """
    expected = op_cls.num_subsystems
    if len(device_operands) != expected:
        raise ValueError(
            f"{op_cls.__name__} expects {expected} device operand(s), "
            f"got {len(device_operands)}"
        )


class _OperationRuleRegistry(Generic[T]):
    """Unconstrained-versus-device-specific rule storage for one rule type.

    An operation family uses at most one mode: `add_unconstrained` and
    `add_device_operands` are mutually exclusive per operation family, the
    same two-mode policy `MatrixImplementationMap` has always documented.

    Additions receive a rule factory rather than an already-wrapped rule so
    operation arity, registration mode, and device-operand arity are all
    validated before family-specific rule wrapping begins. The factory is
    invoked only after those registration checks pass.
    """

    def __init__(self) -> None:
        self._unconstrained: dict[type[Operation], T] = {}
        self._device_operand: dict[type[Operation], dict[DeviceOperands, T]] = {}

    def add_unconstrained(
        self,
        op: Operation | type[Operation],
        rule_factory: Callable[[type[Operation]], T],
    ) -> None:
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._device_operand:
            raise ValueError(
                f"{op_cls.__name__} already has device-specific implementations, "
                "cannot also add an unconstrained implementation for "
                "the same operation. Call remove(op) first if you want "
                "to replace its registrations."
            )
        self._unconstrained[op_cls] = rule_factory(op_cls)

    def add_device_operands(
        self,
        op: Operation | type[Operation],
        device_operands: DeviceOperands,
        rule_factory: Callable[[type[Operation]], T],
    ) -> None:
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        if op_cls in self._unconstrained:
            raise ValueError(
                f"{op_cls.__name__} already has an unconstrained rule "
                "(add); cannot also add a device-specific implementation for "
                "the same operation. Call remove(op) first if you want "
                "to replace its registrations."
            )
        operands = _normalize_device_operands(device_operands)
        _require_device_operands_arity(op_cls, operands)
        rule = rule_factory(op_cls)
        self._device_operand.setdefault(op_cls, {})[operands] = rule

    def get(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> T | None:
        op_cls = _resolve_operation_class(op)
        if device_operands is None:
            return self._unconstrained.get(op_cls)
        table = self._device_operand.get(op_cls)
        if table is not None:
            return table.get(_normalize_device_operands(device_operands))
        return self._unconstrained.get(op_cls)

    def supports(self, op: Operation | type[Operation]) -> bool:
        op_cls = _resolve_operation_class(op)
        return op_cls in self._unconstrained or op_cls in self._device_operand

    def supported_operations(self) -> frozenset[type[Operation]]:
        return frozenset(self._unconstrained | self._device_operand)

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        op_cls = _resolve_operation_class(op)
        return frozenset(self._device_operand.get(op_cls, ()))

    def remove(self, op: Operation | type[Operation]) -> None:
        op_cls = _resolve_operation_class(op)
        self._unconstrained.pop(op_cls, None)
        self._device_operand.pop(op_cls, None)

    def copy(self) -> "_OperationRuleRegistry[T]":
        """Return a new registry with an independent copy of these registrations.

        Rule objects themselves are shared (not deep-copied); rules are
        expected to be immutable or self-contained, so sharing them across
        independent registry copies is safe. The per-operation device-operand
        tables are copied individually (not just the outer dict), so mutating
        one registry's device-specific rules for an operation cannot leak into
        the other's table for that same operation.
        """
        clone: "_OperationRuleRegistry[T]" = _OperationRuleRegistry()
        clone._unconstrained = dict(self._unconstrained)
        clone._device_operand = {
            op_cls: dict(table) for op_cls, table in self._device_operand.items()
        }
        return clone


class _SupportsImplementationLookup(Protocol[T]):
    """Shape every implementation-map family exposes to gate lowering."""

    def supports(self, op: Operation | type[Operation]) -> bool: ...

    def implementation_for(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> T | None: ...


def _select_implementation(
    operation: Operation,
    device_operands: DeviceOperands,
    impl_map: _SupportsImplementationLookup[T],
) -> T:
    """Resolve one operation family's rule for ordered device operands.

    Shared by matrix and pulse gate lowering: both call an implementation map
    exposing `supports`/`implementation_for` in this identical shape, and both
    need the same two distinct `UnsupportedOperationError` reasons - the
    operation family is not registered at all, versus it is registered but not
    for this ordered device-operand key.
    """
    if not impl_map.supports(operation):
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported by this backend"
        )
    rule = impl_map.implementation_for(operation, device_operands=device_operands)
    if rule is None:
        raise UnsupportedOperationError(
            f"{type(operation).__name__} is not supported on device operands {device_operands}"
        )
    return rule
