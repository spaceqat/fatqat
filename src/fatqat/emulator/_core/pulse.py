"""Pulse-plan values: the public authoring surface a pulse implementation
rule returns (``PulseDefinition`` and its building blocks) and registers
under (``PulseImplementationMap``), plus the internal, target-bound
``PulseBlock`` that lowering builds from a definition by attaching one
program occurrence's condition, resolved noise, engine indices, and schedule
position.

``PulseDefinition`` owns model-independent authoring shape. Lowering binds it
once to a gate-capable target, derives target-owned claims, and then constructs
``PulseBlock``. The block is a private already-bound execution value; it
normalizes occurrence fields without repeating definition or target checks.

``PulseImplementationMap`` composes the same private
``implementation._operation_registry`` mechanics ``MatrixImplementationMap`` does,
so pulse and matrix authoring share registration semantics while keeping
distinct rule and result types. ``_invoke_pulse_rule`` is the locked implementation-error
policy: a rule's own ``BackendValidationError`` propagates unchanged, while
any other failure becomes ``PulseImplementationError``.

Nothing here imports a concrete physics model. Lowering asks the bound pulse
target which resources a control implicates and whether its waveform is legal,
then carries those resolved facts into execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

from ..._pulse_values import ControlChannel, PulseControl, TIME_EPSILON
from ...errors import BackendValidationError, PulseImplementationError
from ...implementation._operation_registry import (
    DeviceOperands,
    _OperationRuleRegistry,
)
from ...operations import Operation
from .value_validation import _finite
from .lindblad import ResolvedLindbladTerm
from .target import Frame, ResourceClaim, _PreparedControlBinding


@dataclass(frozen=True)
class PhaseShift:
    """Add an angle to one virtual-frame ledger after a pulse block.

    Attributes:
        frame: Structural frame address returned by ``model.frame(...)``.
        angle_rad: Finite phase increment in radians.
    """

    frame: Frame
    angle_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angle_rad", _finite(self.angle_rad, "phase-shift angle_rad")
        )


@dataclass(frozen=True)
class PhaseSwap:
    """Exchange two virtual-drive frame ledgers after a pulse block.

    This is used by the built-in iSWAP realization so later drive phases stay
    associated with the exchanged excitations.

    Attributes:
        first: First structural frame address.
        second: Distinct second structural frame address.

    Raises:
        BackendValidationError: If both addresses identify the same frame.
    """

    first: Frame
    second: Frame

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise BackendValidationError(
                "phase swap requires two distinct frame references"
            )


FrameAction = PhaseShift | PhaseSwap


def _validate_duration_controls_consistency(
    duration: float, controls: tuple[PulseControl, ...], *, owner: str
) -> None:
    """Zero duration forbids controls; positive duration requires them."""
    if duration == 0.0 and controls:
        raise BackendValidationError(
            f"a zero-duration {owner} cannot contain physical controls"
        )
    if duration > 0.0 and not controls:
        raise BackendValidationError(
            f"a positive-duration {owner} requires physical controls"
        )


def _validate_controls_shape(
    controls: tuple[PulseControl, ...], duration: float, *, owner: str
) -> None:
    """Model-independent control validation shared by both pulse values.

    Checks control type, channel-address type, no implicit same-channel
    summation, and no control sample extending past the enclosing duration.
    Does not touch a target: authoritative channel resolution and waveform
    validation happen once at the bound-target boundary before block
    construction.
    """
    seen_channels: set[ControlChannel] = set()
    for child in controls:
        if not isinstance(child, PulseControl):
            raise BackendValidationError(
                f"{owner} controls must be PulseControl values"
            )
        if not isinstance(child.channel, ControlChannel):
            raise BackendValidationError(
                "pulse control has an unknown channel reference"
            )
        if child.channel in seen_channels:
            raise BackendValidationError(
                f"{owner} cannot implicitly sum controls on one channel"
            )
        seen_channels.add(child.channel)
        if child.start_offset + child.waveform.duration > duration + TIME_EPSILON:
            raise BackendValidationError(
                f"control extends beyond its enclosing {owner}"
            )


def _validate_post_actions_shape(
    post_actions: tuple[FrameAction, ...], *, owner: str
) -> None:
    """Model-independent frame-action validation: type only."""
    for action in post_actions:
        if not isinstance(action, (PhaseShift, PhaseSwap)):
            raise BackendValidationError(f"{owner} has an unknown frame action")


@dataclass(frozen=True)
class PulseDefinition:
    """One reusable physical pulse recipe, independent of any occurrence.

    Returned by a pulse implementation rule registered with
    :class:`~fatqat.emulator.PulseImplementationMap`. Contains only the
    physical realization: duration, sampled
    controls and any post-block frame actions. It carries no target-owned
    resource claims, classical condition, resolved noise, engine
    index, or schedule position - those are one lowered program occurrence's
    facts, attached by ``emulator._core.planning._lower_gate`` when it converts a
    definition into a target-bound ``PulseBlock``.

    ``duration``, every :class:`~fatqat.emulator.SampledWaveform` time grid,
    and every :class:`~fatqat.emulator.PulseControl` ``start_offset`` use the
    owning model's native time coordinate; this type does not claim
    nanoseconds or any other unit.

    A zero-duration definition represents a virtual operation and must not
    contain controls. A positive-duration definition must contain at least one
    control. Every driven channel must fit inside ``duration``. Target-owned
    scheduling claims are derived later when an occurrence binds to a target.

    Attributes:
        duration: Non-negative block duration in ``model.time_unit``.
        controls: Sampled physical controls. Duplicate channels are rejected;
            explicitly sum contributions before constructing the definition.
        post_actions: Virtual :class:`PhaseShift` or :class:`PhaseSwap`
            actions applied after the physical interval.

    Raises:
        BackendValidationError: If the duration, controls, or frame actions
            are structurally inconsistent.
    """

    duration: float
    controls: tuple[PulseControl, ...]
    post_actions: tuple[FrameAction, ...] = ()

    def __post_init__(self) -> None:
        duration = _finite(self.duration, "pulse-definition duration", nonnegative=True)
        _validate_duration_controls_consistency(
            duration, self.controls, owner="pulse definition"
        )
        _validate_controls_shape(self.controls, duration, owner="pulse definition")
        _validate_post_actions_shape(self.post_actions, owner="pulse definition")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "post_actions", tuple(self.post_actions))


@dataclass(frozen=True)
class PulseBlock:
    """One already-bound atomic pulse occurrence on its target's time axis."""

    duration: float
    controls: tuple[PulseControl, ...]
    control_bindings: tuple[_PreparedControlBinding, ...]
    resource_claims: tuple[ResourceClaim, ...]
    post_actions: tuple[FrameAction, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None
    start_time: float | None = None
    noise: tuple[ResolvedLindbladTerm, ...] = ()
    target_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.condition is not None:
            normalized = tuple(self.condition)
            if not normalized or any(
                type(clbit) is not int
                or clbit < 0
                or type(value) is not int
                or value < 0
                for clbit, value in normalized
            ):
                raise BackendValidationError(
                    "pulse-block condition must contain non-negative integer terms"
                )
            object.__setattr__(self, "condition", normalized)
        if self.start_time is not None:
            object.__setattr__(
                self,
                "start_time",
                _finite(self.start_time, "pulse-block start_time", nonnegative=True),
            )
        object.__setattr__(self, "controls", tuple(self.controls))
        bindings = tuple(self.control_bindings)
        if len(bindings) != len(self.controls) or any(
            not isinstance(binding, _PreparedControlBinding) for binding in bindings
        ):
            raise BackendValidationError(
                "pulse block requires one control binding per control"
            )
        object.__setattr__(self, "control_bindings", bindings)
        object.__setattr__(self, "resource_claims", tuple(self.resource_claims))
        object.__setattr__(self, "post_actions", tuple(self.post_actions))
        object.__setattr__(self, "noise", tuple(self.noise))
        if self.target_indices is not None:
            target_indices = tuple(self.target_indices)
            if (
                not target_indices
                or len(set(target_indices)) != len(target_indices)
                or any(type(index) is not int or index < 0 for index in target_indices)
            ):
                raise BackendValidationError(
                    "pulse-block target indices must be distinct non-negative ints"
                )
            object.__setattr__(self, "target_indices", target_indices)


class _PulseImplementation:
    """Base class for a pulse-family implementation rule.

    Every stored implementation has one internal invocation shape. The
    wrapper remembers whether the authored callable explicitly requests the
    selected ordered ``device_operands`` tuple.
    """

    wants_device_operands: bool = False

    def __call__(
        self,
        operation: Operation,
        *,
        device_operands: DeviceOperands,
    ) -> PulseDefinition:
        raise NotImplementedError


class _CallablePulseImplementation(_PulseImplementation):
    """Adapt a pulse callable to the single internal invocation shape."""

    def __init__(self, func: Callable, wants_device_operands: bool) -> None:
        self._func = func
        self.wants_device_operands = wants_device_operands

    def __call__(
        self,
        operation: Operation,
        *,
        device_operands: DeviceOperands,
    ) -> PulseDefinition:
        if self.wants_device_operands:
            return self._func(operation, device_operands=device_operands)
        return self._func(operation)


class _FixedPulseImplementation(_PulseImplementation):
    """Return one already-built pulse definition for every invocation."""

    def __init__(self, definition: PulseDefinition) -> None:
        self._definition = definition

    def __call__(
        self,
        operation: Operation,
        *,
        device_operands: DeviceOperands,
    ) -> PulseDefinition:
        del operation, device_operands
        return self._definition


def _callable_wants_device_operands(rule: Callable) -> bool:
    """Return whether ``rule`` explicitly accepts ``device_operands=``.

    ``**kwargs`` alone is intentionally insufficient: an unconstrained pulse
    rule must state that it understands physical device addresses. A
    positional-only parameter also cannot accept the keyword used by the map.
    Uninspectable callables are conservatively operand-unaware.
    """
    try:
        parameter = inspect.signature(rule).parameters.get("device_operands")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _wrap_pulse_rule(
    op_cls: type[Operation],
    rule: "_PulseImplementation | Callable | PulseDefinition",
) -> _PulseImplementation:
    """Normalize an `add()` implementation argument into a `_PulseImplementation`.

    A concrete :class:`PulseDefinition` becomes a fixed implementation. Every
    callable, including a callable implementation object, is wrapped while the
    pulse-only signature detector records whether it explicitly accepts
    ``device_operands=``. Every stored value therefore has one internal call
    shape, while wrong callable arity remains deferred until first use through
    :func:`_invoke_pulse_rule`.

    Raises:
        TypeError: If `rule` is neither a `PulseDefinition` nor callable,
            checked explicitly here so the error names the operation and bad
            value.
    """
    if isinstance(rule, PulseDefinition):
        return _FixedPulseImplementation(rule)
    if not callable(rule):
        raise TypeError(
            f"rule for {op_cls.__name__} must be a PulseDefinition, pulse "
            f"implementation object, or callable, got {rule!r}"
        )
    return _CallablePulseImplementation(rule, _callable_wants_device_operands(rule))


def _invoke_pulse_rule(
    rule: _PulseImplementation,
    operation: Operation,
    *,
    device_operands: DeviceOperands,
) -> PulseDefinition:
    """Call a selected pulse rule and enforce the locked implementation-error policy.

    A rule's own `BackendValidationError` (including `UnsupportedOperationError`)
    propagates unchanged: that is the rule's deliberate validation, not an
    implementation defect. Any other exception the rule raises, or a
    non-`PulseDefinition` return value, becomes `PulseImplementationError`
    naming the operation, with the original exception preserved as
    `__cause__` when one exists.
    """
    try:
        result = rule(operation, device_operands=device_operands)
    except BackendValidationError:
        raise
    except Exception as exc:
        raise PulseImplementationError(
            f"implementation for {type(operation).__name__} raised: {exc}"
        ) from exc
    if not isinstance(result, PulseDefinition):
        raise PulseImplementationError(
            f"implementation for {type(operation).__name__} returned "
            f"{result!r}, expected a PulseDefinition"
        )
    return result


class PulseImplementationMap:
    """Resolve operation families and device operands to pulse implementations.

    This value maps ordinary gate operations to reusable pulse definitions.
    Channel-addressed :class:`~fatqat.operations.PulseOperation` values bypass
    it: the selected emulator resolves each
    :attr:`~fatqat.emulator.PulseControl.channel` directly against its physical
    model. A family may provide an empty built-in map when it has no standard
    gate realizations, while the same general map path remains available for
    user-supplied rules.

    It follows the registration and copy rules of
    :class:`~fatqat.implementation.MatrixImplementationMap`, but its rules
    return :class:`PulseDefinition` values. Invalid results and unexpected
    rule failures raise :exc:`~fatqat.errors.PulseImplementationError`.

    An operand-aware rule has the signature
    ``rule(operation, *, device_operands) -> PulseDefinition``. The tuple is
    the exact ordered physical address used for map selection. Fixed
    definitions and operand-unaware callables require an explicit
    ``device_operands=`` registration.

    Use :func:`~fatqat.emulator.default_transmon_gate_implementation_map`
    as the starting point when replacing one built-in realization. A
    :class:`~fatqat.emulator.TransmonEmulator` copies the map passed to its
    constructor, so later registration changes do not alter that backend.

    Examples:
        Replace one unconstrained implementation while retaining the other
        defaults::

            implementations = default_transmon_gate_implementation_map(
                model=model,
                calibration=calibration,
            )
            implementations.remove(ops.CZ)
            implementations.add(ops.CZ, custom_cz)
    """

    def __init__(self) -> None:
        self._registry: _OperationRuleRegistry[_PulseImplementation] = (
            _OperationRuleRegistry()
        )

    def add(
        self,
        op: Operation | type[Operation],
        implementation: Callable[..., PulseDefinition] | PulseDefinition,
        *,
        device_operands: DeviceOperands | None = None,
    ) -> None:
        """Add an unconstrained or device-specific pulse implementation.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance (e.g.
                ``ops.CZ``) or subclass. Normalized to the operation's class for
                the registry key.
            implementation: A concrete :class:`PulseDefinition`, a callable
                accepting ``operation`` and optionally an explicit keyword
                ``device_operands``, or a callable implementation object.
            device_operands: An ordered hashable tuple identifying the
                device-level target this rule is restricted to. Omit for an
                unconstrained rule that applies to every legal target of the
                operation's arity.

        Raises:
            TypeError: If ``op`` is neither an :py:class:`~fatqat.operations.Operation`
                instance nor subclass, if its operation class has variable
                arity, or if ``implementation`` is not callable.
            ValueError: If ``device_operands``' length does not match the
                operation's arity, if an operand-unaware implementation is
                registered without explicit ``device_operands``, or if ``op``
                already has a registration in the other mode; see
                :meth:`~fatqat.implementation.MatrixImplementationMap.add` for
                why the two modes are mutually exclusive.
        """
        if device_operands is not None:
            self._registry.add_device_operands(
                op,
                device_operands,
                lambda op_cls: _wrap_pulse_rule(op_cls, implementation),
            )
            return

        def unconstrained(op_cls: type[Operation]) -> _PulseImplementation:
            wrapped = _wrap_pulse_rule(op_cls, implementation)
            if not wrapped.wants_device_operands:
                raise ValueError(
                    f"operand-unaware pulse implementation for {op_cls.__name__} "
                    "requires explicit device_operands"
                )
            return wrapped

        self._registry.add_unconstrained(op, unconstrained)

    def supports(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> bool:
        """Return whether this map has any rule for the operation family.

        Same semantics as
        :meth:`~fatqat.implementation.MatrixImplementationMap.supports`.
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
    ) -> Callable[..., PulseDefinition] | None:
        """Return the pulse implementation selected for an operation.

        Same lookup semantics as
        :meth:`~fatqat.implementation.MatrixImplementationMap.implementation_for`:
        with ``device_operands`` omitted, only the unconstrained rule is
        consulted; with ``device_operands`` given, a family with
        device-specific rules consults only those, while a family with only an
        unconstrained rule returns it for every device operands.
        """
        return self._registry.get(op, device_operands=device_operands)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return every operation family with at least one implementation."""
        return self._registry.supported_operations()

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        """Return the finite set of device operands selected for an operation.

        Same semantics as
        :meth:`~fatqat.implementation.MatrixImplementationMap.device_operands_for`.
        """
        return self._registry.device_operands_for(op)

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove a registered pulse implementation, if present.

        Removing an operation that was never registered is a no-op.
        """
        self._registry.remove(op)

    def copy(self) -> "PulseImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects are shared (not deep-copied) between the original and the
        copy, matching
        :meth:`~fatqat.implementation.MatrixImplementationMap.copy`. Later
        mutations of either map's registrations never affect the other.
        """
        clone = PulseImplementationMap()
        clone._registry = self._registry.copy()
        return clone
