"""Pulse definitions, implementation maps, and internal plan values."""

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
    """Shift one virtual frame after an ordinary gate's pulse interval.

    Obtain the frame from a compatible emulator model. The named resource must
    exist in the model used to run the program.

    Attributes:
        frame: Frame returned by ``model.frame(...)``.
        angle_rad: Finite phase increment in radians.

    Raises:
        BackendValidationError: If ``angle_rad`` is not a finite ``int`` or
            ``float``, excluding booleans.
    """

    frame: Frame
    angle_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angle_rad", _finite(self.angle_rad, "phase-shift angle_rad")
        )


@dataclass(frozen=True)
class PhaseSwap:
    """Exchange two virtual frames after an ordinary gate's pulse interval.

    Obtain both frames from a compatible emulator model. Exchanging them keeps
    later drive phases associated with the exchanged excitations.

    Attributes:
        first: First frame returned by ``model.frame(...)``.
        second: Distinct second frame.

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
    """Describe the pulses used to implement an ordinary gate.

    A ``PulseImplementationMap`` rule returns a definition containing the
    duration, controls, and optional frame actions for an ordinary operation.

    All times use the model's time unit. A zero-duration definition may contain
    frame actions but no controls. A positive-duration definition needs at
    least one control. Channels must be unique, and every shifted waveform must
    fit within ``duration``. The emulator checks channel and model constraints
    when the definition is used.

    Args:
        duration: Finite non-negative duration in the model's time unit.
        controls: Controls for the physical interval.
        post_actions: Frame actions applied after the interval.

    Raises:
        BackendValidationError: If the duration, controls, or frame actions
            are invalid or inconsistent.
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
            f"rule for {op_cls.__name__} must be a PulseDefinition or callable, "
            f"got {rule!r}"
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
    """Map ordinary gates to pulse definitions.

    Direct ``PulseOperation`` values bypass this map because their controls
    already name physical channels.

    A device-specific rule applies to one exact ``device_operands`` tuple; a
    general rule applies to every tuple. These are ordered physical labels,
    not program register references. See ``add()`` for the accepted rule
    forms.

    When a rule is used, its ``BackendValidationError`` is reported unchanged.
    Other exceptions and return values other than ``PulseDefinition`` are
    reported as ``PulseImplementationError``.

    Examples:
        Register a fixed definition for one physical operand:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> model = fq.emulator.TransmonModel.from_document(
        ...     fq.emulator.load_model_document("transmon.reference")
        ... )
        >>> waveform = fq.emulator.SampledWaveform(
        ...     (0.0, 10.0, 20.0), (0.0, 0.02, 0.0)
        ... )
        >>> control = fq.emulator.PulseControl(
        ...     model.control.drive("q0"), waveform
        ... )
        >>> definition = fq.emulator.PulseDefinition(20.0, (control,))
        >>> implementations = fq.emulator.PulseImplementationMap()
        >>> implementations.add(
        ...     ops.X, definition, device_operands=("q0",)
        ... )
        >>> implementations.supports(ops.X, device_operands=("q0",))
        True
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
        """Add a general or device-specific pulse implementation.

        With explicit ``device_operands``, use a fixed ``PulseDefinition`` or
        a callable that accepts the operation and may also accept
        ``device_operands`` by keyword. Without explicit operands, the callable
        must accept ``device_operands`` by keyword.
        An operation family cannot mix general and device-specific rules;
        remove it before switching forms. Adding the same family or operand
        tuple again replaces its previous rule.

        Args:
            op: Operation instance or class to implement.
            implementation: A ``PulseDefinition`` or a callable that returns
                one. When ``device_operands`` is omitted, the callable must
                explicitly accept that keyword.
            device_operands: Ordered physical labels for a device-specific
                rule. The tuple length must match the operation's arity. Omit
                this argument for a rule that applies to every tuple.

        Raises:
            TypeError: If ``op`` is neither an ``Operation`` instance nor
                subclass; if its operation class has variable arity or is a
                ``PulseOperation``; if explicit operands are invalid; or if
                ``implementation`` is neither a ``PulseDefinition`` nor
                callable.
            ValueError: If ``device_operands`` length does not match the
                operation's arity; if operands are omitted for a fixed
                definition or callable that does not explicitly accept
                ``device_operands``; or if ``op`` already uses the other
                registration form.
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
        """Return whether this map has a matching rule.

        Without ``device_operands``, this reports whether the family has any
        rule. With operands, it checks the exact tuple or a general rule.

        Args:
            op: Operation instance or class to query.
            device_operands: Optional ordered tuple of physical labels.

        Raises:
            TypeError: If ``op`` or explicit operands are invalid.
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
        """Return the pulse implementation for an operation.

        Without ``device_operands``, this returns only a general rule. With
        operands, it returns the rule for that tuple or a general rule.

        Args:
            op: Operation instance or class to query.
            device_operands: Optional ordered tuple of physical labels.

        Returns:
            The matching callable, or ``None``.

        Raises:
            TypeError: If ``op`` or explicit operands are invalid.
        """
        return self._registry.get(op, device_operands=device_operands)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return the operation classes with at least one implementation."""
        return self._registry.supported_operations()

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        """Return the device-specific operand tuples for an operation.

        The empty set means either that the family is unregistered or that it
        has one general rule. Use ``supports(op)`` to distinguish those
        cases.

        Args:
            op: Operation instance or class to query.

        Raises:
            TypeError: If ``op`` is not an operation instance or subclass.
        """
        return self._registry.device_operands_for(op)

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove a registered pulse implementation, if present.

        Removing an operation that was never registered is a no-op.

        Args:
            op: Operation instance or subclass whose entire family
                registration is removed.

        Raises:
            TypeError: If ``op`` is not an operation instance or subclass.
        """
        self._registry.remove(op)

    def copy(self) -> "PulseImplementationMap":
        """Return a copy whose registrations can be changed independently."""
        clone = PulseImplementationMap()
        clone._registry = self._registry.copy()
        return clone
