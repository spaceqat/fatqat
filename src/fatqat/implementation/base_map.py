"""Shared registration storage and lookup for implementation maps."""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import TYPE_CHECKING, Callable, Generic, Self, TypeVar

from ..errors import UnsupportedOperationError
from ..operations import Operation
from ..resource_layout import DeviceOperand

if TYPE_CHECKING:
    from ..noise.base import Channel


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


def _reject_direct_control(op_cls: type[Operation] | type[Channel]) -> None:
    """Direct controls already define their behavior and cannot be registered."""
    if issubclass(op_cls, Operation) and op_cls._is_direct_control:
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
    op_cls: type[Operation] | type[Channel], device_operands: DeviceOperands
) -> None:
    """Raise `ValueError` if device operands do not match a declared fixed arity.

    Only arity is checked here: the general registry does not know what a
    target key element means (an integer device label, a zone name, ...), so
    it cannot range-check or type-check individual elements. That is left to
    the backend that constructs a device-specific map.
    """
    # Channels may determine their width through an instance property.
    expected = op_cls.num_subsystems
    if isinstance(expected, int) and len(device_operands) != expected:
        raise ValueError(
            f"{op_cls.__name__} expects {expected} device operand(s), "
            f"got {len(device_operands)}"
        )


class _BaseImplementationMap(Generic[T]):
    """Unconstrained-versus-device-specific rule storage for one rule type.

    An operation family uses at most one mode: `_add_unconstrained` and
    `_add_device_operands` are mutually exclusive per operation family, the
    same two-mode policy used by matrix, channel, and pulse rules.

    Additions receive a rule factory rather than an already-wrapped rule so
    direct-control exclusion, registration mode, and device-operand arity are all
    validated before family-specific rule wrapping begins. The factory is
    invoked only after those registration checks pass.
    """

    def __init__(
        self,
        resolve_class: Callable[
            [Operation | Channel | type[Operation] | type[Channel]],
            type[Operation] | type[Channel],
        ] = _resolve_operation_class,
    ) -> None:
        self._resolve_class = resolve_class
        self._unconstrained: dict[type[Operation] | type[Channel], T] = {}
        self._device_operand: dict[
            type[Operation] | type[Channel], dict[DeviceOperands, T]
        ] = {}

    def _add_unconstrained(
        self,
        op: Operation | Channel | type[Operation] | type[Channel],
        rule_factory: Callable[[type[Operation] | type[Channel]], T],
    ) -> None:
        op_cls = self._resolve_class(op)
        _reject_direct_control(op_cls)
        if op_cls in self._device_operand:
            raise ValueError(
                f"{op_cls.__name__} already has device-specific implementations, "
                "cannot also add an unconstrained implementation for "
                "the same operation. Call remove(op) first if you want "
                "to replace its registrations."
            )
        self._unconstrained[op_cls] = rule_factory(op_cls)

    def _add_device_operands(
        self,
        op: Operation | Channel | type[Operation] | type[Channel],
        device_operands: DeviceOperands,
        rule_factory: Callable[[type[Operation] | type[Channel]], T],
    ) -> None:
        op_cls = self._resolve_class(op)
        _reject_direct_control(op_cls)
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

    def implementation_for(
        self,
        op: Operation | Channel | type[Operation] | type[Channel],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> T | None:
        """Return the matching implementation, or ``None`` if none is registered.

        Lookup uses the exact concrete class; a base-class registration does not
        cover subclasses. See the map class for accepted key types.

        Without ``device_operands``, return only a uniform rule. With operands,
        return the exact device-specific rule, or the family's uniform rule
        when it uses uniform registration.
        """
        op_cls = self._resolve_class(op)
        if device_operands is None:
            return self._unconstrained.get(op_cls)
        table = self._device_operand.get(op_cls)
        if table is not None:
            return table.get(_normalize_device_operands(device_operands))
        return self._unconstrained.get(op_cls)

    def supports(
        self,
        op: Operation | Channel | type[Operation] | type[Channel],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> bool:
        """Return whether a matching implementation is registered.

        Without ``device_operands``, report whether the family has any rule.
        With operands, check the exact tuple or a uniform rule.
        """
        if device_operands is not None:
            return (
                self.implementation_for(op, device_operands=device_operands) is not None
            )
        op_cls = self._resolve_class(op)
        return op_cls in self._unconstrained or op_cls in self._device_operand

    def _registered_types(self) -> frozenset[type[Operation] | type[Channel]]:
        return frozenset(self._unconstrained | self._device_operand)

    def device_operands_for(
        self, op: Operation | Channel | type[Operation] | type[Channel]
    ) -> frozenset[DeviceOperands]:
        """Return the explicitly registered device tuples for this family.

        An empty set means the family is unregistered or uses a uniform rule.
        """
        op_cls = self._resolve_class(op)
        return frozenset(self._device_operand.get(op_cls, ()))

    def remove(self, op: Operation | Channel | type[Operation] | type[Channel]) -> None:
        """Remove all rules for this family; missing registrations are ignored."""
        op_cls = self._resolve_class(op)
        self._unconstrained.pop(op_cls, None)
        self._device_operand.pop(op_cls, None)

    def copy(self) -> Self:
        """Return an independently editable copy, preserving the map's subclass.

        Adding or removing rules on either map does not affect the other.
        Registered implementation objects are shared rather than deep-copied.
        """
        # Preserve family metadata without rerunning subclass constructors.
        clone = shallow_copy(self)
        clone._unconstrained = dict(self._unconstrained)
        clone._device_operand = {
            op_cls: dict(table) for op_cls, table in self._device_operand.items()
        }
        return clone


def _select_implementation(
    operation: Operation | Channel,
    device_operands: DeviceOperands,
    impl_map: _BaseImplementationMap[T],
) -> T:
    """Resolve one operation family's rule for ordered device operands.

    Shared by matrix, channel, and pulse lowering: each calls an implementation map
    exposing `supports`/`implementation_for` in this identical shape. They all
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
