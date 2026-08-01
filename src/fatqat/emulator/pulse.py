"""Pulse-plan values: the public authoring surface a pulse implementation
rule returns (``PulseDefinition`` and its building blocks) and registers
under (``PulseImplementationMap``), plus the internal, model-owned
``PulseBlock`` that lowering builds from a definition by attaching one
program occurrence's condition, resolved noise, engine indices, and schedule
position.

``PulseDefinition`` and ``PulseBlock`` share the model-independent structural
checks (duration, control shape, resource-claim shape, frame-action shape)
through the module-level ``_validate_*`` helpers below, so the two never
diverge on what counts as a well-formed pulse. Only ``PulseBlock`` also
performs the model-bound checks (channel/resource/coupling/frame binding,
driven-control claim coverage), because only it carries a ``PhysicsModel``.

``PulseImplementationMap`` composes the same private
``implementation._operation_registry`` mechanics ``MatrixImplementationMap`` does,
so the two families share registration semantics while keeping distinct rule
and result types. ``_invoke_pulse_rule`` is the locked implementation-error
policy: a rule's own ``BackendValidationError`` propagates unchanged, while
any other failure becomes ``PulseImplementationError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import numpy as np

from ..errors import BackendValidationError, PulseImplementationError
from ..implementation._operation_registry import (
    DeviceOperands,
    _OperationRuleRegistry,
)
from ..operations import Operation
from .lindblad import ResolvedLindbladTerm
from .superconducting import (
    CalibrationSpec,
    ControlChannelRef,
    CouplingRef,
    FrameRef,
    PhysicsModel,
    SubsystemResourceRef,
)


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    """Normalize one finite scalar, optionally requiring non-negativity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendValidationError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value) or (nonnegative and value < 0):
        raise BackendValidationError(f"{name} must be finite and non-negative")
    return value


def _freeze(values: Any, *, dtype: type = complex) -> np.ndarray:
    """Copy values to a read-only NumPy array of the requested dtype."""
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SampledControl:
    """One sampled physical control on a model-minted channel.

    A control owns a local sample grid. Its samples are shifted by
    ``start_offset`` inside the enclosing :class:`PulseDefinition`; the
    implementation rule does not need to convert them to absolute program
    time. The owning model defines the time unit. For the built-in transmon
    model it is ``model.time_unit == "ns"``.

    ``coefficients`` may be complex for a drive channel, where real and
    imaginary components encode the two quadratures. Detuning and exchange
    controls must be real and are checked when the private solver adapter
    binds the definition. Arrays are copied and made read-only.

    Attributes:
        channel: Opaque control-channel handle returned by ``model``.
        tlist: One-dimensional, strictly increasing local sample times. The
            first value must be zero and at least two samples are required.
        coefficients: Complex sample values aligned one-to-one with ``tlist``.
        start_offset: Non-negative offset from the enclosing pulse's start.

    Raises:
        BackendValidationError: If the offset or samples are non-finite, the
            arrays are not aligned one-dimensional arrays, or ``tlist`` does
            not start at zero and increase strictly.
    """

    channel: ControlChannelRef
    tlist: np.ndarray
    coefficients: np.ndarray
    start_offset: float = 0.0

    def __post_init__(self) -> None:
        start_offset = _finite(
            self.start_offset, "control start_offset", nonnegative=True
        )
        tlist = np.asarray(self.tlist, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=complex)
        if tlist.ndim != 1 or coefficients.ndim != 1 or len(tlist) != len(coefficients):
            raise BackendValidationError(
                "control tlist and coefficients must be matching one-dimensional arrays"
            )
        if (
            len(tlist) < 2
            or not np.all(np.isfinite(tlist))
            or not np.all(np.isfinite(coefficients))
        ):
            raise BackendValidationError(
                "control samples must be finite and contain at least two points"
            )
        if tlist[0] != 0.0 or np.any(np.diff(tlist) <= 0.0):
            raise BackendValidationError(
                "control tlist must start at zero and be strictly increasing"
            )
        object.__setattr__(self, "start_offset", start_offset)
        object.__setattr__(self, "tlist", _freeze(tlist, dtype=float))
        object.__setattr__(self, "coefficients", _freeze(coefficients))

    @property
    def duration(self) -> float:
        """Return the local control duration in the model's time unit."""
        return float(self.tlist[-1])


@dataclass(frozen=True)
class PhaseShift:
    """Add an angle to one virtual-frame ledger after a pulse block.

    Attributes:
        frame: Opaque frame handle returned by ``model.frame(...)``.
        angle_rad: Finite phase increment in radians.
    """

    frame: FrameRef
    angle_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angle_rad", _finite(self.angle_rad, "phase-shift angle_rad")
        )


@dataclass(frozen=True)
class PhaseSwap:
    """Exchange two virtual-drive frame ledgers after a pulse block.

    This is used by the built-in iSWAP realization so later drive phases stay
    associated with the exchanged logical excitations.

    Attributes:
        first: First model-minted frame handle.
        second: Distinct second model-minted frame handle.

    Raises:
        BackendValidationError: If both handles identify the same frame.
    """

    first: FrameRef
    second: FrameRef

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise BackendValidationError(
                "phase swap requires two distinct frame references"
            )


FrameAction = PhaseShift | PhaseSwap
ResourceClaim = SubsystemResourceRef | CouplingRef


def _validate_duration_controls_consistency(
    duration: float, controls: tuple[SampledControl, ...], *, owner: str
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
    controls: tuple[SampledControl, ...], duration: float, *, owner: str
) -> None:
    """Model-independent control validation shared by both pulse values.

    Checks control type, channel-reference type, no implicit same-channel
    summation, and no control sample extending past the enclosing duration.
    Does not touch the model: whether a channel actually resolves on a given
    model is a `PulseBlock`-only, model-bound check.
    """
    seen_channels: set[ControlChannelRef] = set()
    for child in controls:
        if not isinstance(child, SampledControl):
            raise BackendValidationError(
                f"{owner} controls must be SampledControl values"
            )
        if not isinstance(child.channel, ControlChannelRef):
            raise BackendValidationError(
                "pulse control has an unknown channel reference"
            )
        if child.channel in seen_channels:
            raise BackendValidationError(
                f"{owner} cannot implicitly sum controls on one channel"
            )
        seen_channels.add(child.channel)
        if child.start_offset + child.duration > duration + 1e-12:
            raise BackendValidationError(
                f"control extends beyond its enclosing {owner}"
            )


def _validate_resource_claims_shape(
    resource_claims: tuple[ResourceClaim, ...], *, owner: str
) -> None:
    """Model-independent resource-claim validation: non-empty, type, no duplicates."""
    if not resource_claims:
        raise BackendValidationError(f"{owner} must claim at least one model resource")
    seen: set[ResourceClaim] = set()
    for resource in resource_claims:
        if not isinstance(resource, (SubsystemResourceRef, CouplingRef)):
            raise BackendValidationError(f"{owner} has an unknown resource claim")
        if resource in seen:
            raise BackendValidationError(f"{owner} has a duplicate resource claim")
        seen.add(resource)


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

    Returned by a pulse implementation rule (see the pulse implementation
    map). Contains only the physical realization: duration, sampled
    controls, the model resources/couplings it claims, and any post-block
    frame actions. It carries no classical condition, resolved noise, engine
    index, or schedule position - those are one lowered program occurrence's
    facts, attached by ``emulator.planning._lower_gate`` when it converts a
    definition into a model-owned `PulseBlock`.

    ``duration``, and every :class:`SampledControl`'s ``tlist`` and
    ``start_offset``, use the
    owning model's native time coordinate; this type does not claim
    nanoseconds or any other unit.

    A zero-duration definition represents a virtual operation and must not
    contain controls. A positive-duration definition must contain at least one
    control. Every driven channel must fit inside ``duration``. Resource
    claims are mandatory even for virtual operations because they define
    ordering and exclusion during lightweight scheduling.

    Attributes:
        duration: Non-negative block duration in ``model.time_unit``.
        controls: Sampled physical controls. Duplicate channels are rejected;
            explicitly sum contributions before constructing the definition.
        resource_claims: Model subsystem and/or coupling handles reserved for
            the complete block.
        post_actions: Virtual :class:`PhaseShift` or :class:`PhaseSwap`
            actions applied after the physical interval.

    Raises:
        BackendValidationError: If the duration, control shapes, resource
            claims, or frame actions are structurally inconsistent.
    """

    duration: float
    controls: tuple[SampledControl, ...]
    resource_claims: tuple[ResourceClaim, ...]
    post_actions: tuple[FrameAction, ...] = ()

    def __post_init__(self) -> None:
        duration = _finite(self.duration, "pulse-definition duration", nonnegative=True)
        _validate_resource_claims_shape(self.resource_claims, owner="pulse definition")
        _validate_duration_controls_consistency(
            duration, self.controls, owner="pulse definition"
        )
        _validate_controls_shape(self.controls, duration, owner="pulse definition")
        _validate_post_actions_shape(self.post_actions, owner="pulse definition")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "resource_claims", tuple(self.resource_claims))
        object.__setattr__(self, "post_actions", tuple(self.post_actions))


@dataclass(frozen=True)
class PulseBlock:
    """One atomic model-owned pulse block on its model's native time axis."""

    model: PhysicsModel
    duration: float
    controls: tuple[SampledControl, ...]
    resource_claims: tuple[ResourceClaim, ...]
    post_actions: tuple[FrameAction, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None
    start_time: float | None = None
    noise: tuple[ResolvedLindbladTerm, ...] = ()
    target_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        duration = _finite(self.duration, "pulse-block duration", nonnegative=True)
        _validate_resource_claims_shape(self.resource_claims, owner="pulse block")
        _validate_duration_controls_consistency(
            duration, self.controls, owner="pulse block"
        )
        _validate_controls_shape(self.controls, duration, owner="pulse block")
        _validate_post_actions_shape(self.post_actions, owner="pulse block")

        required_claim_sets: list[set[ResourceClaim]] = []
        for child in self.controls:
            control_ordinal = self.model.bind_control(child.channel)
            if child.channel.kind == "exchange":
                coupling = self.model.couplings[control_ordinal]
                required_claim_sets.append(
                    {
                        self.model.resource(subsystem_id)
                        for subsystem_id in coupling.subsystem_ids
                    }
                    | {self.model.coupling(*coupling.subsystem_ids)}
                )
            else:
                required_claim_sets.append(
                    {self.model.resource(self.model.subsystem_ids[control_ordinal])}
                )

        seen_resources: set[ResourceClaim] = set()
        for resource in self.resource_claims:
            if isinstance(resource, SubsystemResourceRef):
                self.model.bind_resource(resource)
            else:
                self.model.bind_coupling(resource)
            seen_resources.add(resource)
        for required_claims in required_claim_sets:
            if not required_claims <= seen_resources:
                raise BackendValidationError(
                    "pulse block resource claims do not cover a driven control"
                )

        for action in self.post_actions:
            if isinstance(action, PhaseShift):
                self.model.bind_frame(action.frame)
            else:
                self.model.bind_frame(action.first)
                self.model.bind_frame(action.second)

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
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "controls", tuple(self.controls))
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

    A rule receives the applied :py:class:`~fatqat.operations.Operation`
    instance, the ordered physical-model resource handles corresponding to
    its program targets, the immutable owning `PhysicsModel`, and the
    `CalibrationSpec`, and returns one reusable `PulseDefinition`. Most
    callers never need to subclass this directly: `PulseImplementationMap.add`
    auto-wraps a bare callable with this exact signature. Subclass and
    override `__call__` for a stateful or configured implementation.
    """

    def __call__(
        self,
        operation: Operation,
        *,
        targets: tuple[Any, ...],
        model: PhysicsModel,
        calibration: CalibrationSpec,
    ) -> PulseDefinition:
        raise NotImplementedError


class _CallablePulseImplementation(_PulseImplementation):
    """Adapts a bare `f(operation, *, targets, model, calibration)` callable."""

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self,
        operation: Operation,
        *,
        targets: tuple[Any, ...],
        model: PhysicsModel,
        calibration: CalibrationSpec,
    ) -> PulseDefinition:
        return self._func(
            operation, targets=targets, model=model, calibration=calibration
        )


def _wrap_pulse_rule(
    op_cls: type[Operation], rule: "_PulseImplementation | Callable"
) -> _PulseImplementation:
    """Normalize an `add()` implementation argument into a `_PulseImplementation`.

    Accepts an already-built `_PulseImplementation` (returned as-is) or a bare
    `f(operation, *, targets, model, calibration)` callable (wrapped). Every
    stored rule is a `_PulseImplementation` instance, so `implementation_for()`
    always returns a uniform type regardless of how the rule was registered -
    the pulse-map analog of `MatrixImplementationMap`'s uniform `MatrixImplementation`
    return.

    Unlike the matrix map, a pulse rule has exactly one accepted callable
    shape, so there is no signature-detection step: a callable of the wrong
    shape is not rejected here; it fails on first use (see
    `_invoke_pulse_rule`), the same deferred-failure contract the matrix map
    uses for a bare callable.

    Raises:
        TypeError: If `rule` is neither a `_PulseImplementation` nor a
            callable, checked explicitly here so the error names the
            operation and the bad value.
    """
    if isinstance(rule, _PulseImplementation):
        return rule
    if not callable(rule):
        raise TypeError(
            f"rule for {op_cls.__name__} must be a pulse implementation object "
            f"or callable, got {rule!r}"
        )
    return _CallablePulseImplementation(rule)


def _invoke_pulse_rule(
    rule: _PulseImplementation,
    operation: Operation,
    *,
    targets: tuple[Any, ...],
    model: PhysicsModel,
    calibration: CalibrationSpec,
) -> PulseDefinition:
    """Call a selected pulse rule and enforce the locked implementation-error policy.

    A rule's own `BackendValidationError` (including `UnsupportedOperationError`)
    propagates unchanged: that is the rule's deliberate validation, not an
    implementation defect (e.g. the default CZ rule's target-orientation or
    missing-edge-recipe failures). Any other exception the rule raises, or a
    non-`PulseDefinition` return value, becomes `PulseImplementationError`
    naming the operation, with the original exception preserved as
    `__cause__` when one exists.
    """
    try:
        result = rule(operation, targets=targets, model=model, calibration=calibration)
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

    Structurally identical to :class:`~fatqat.implementation.MatrixImplementationMap`
    - same instance/class
    normalization, same mutually exclusive unconstrained-versus-device-
    specific registration policy, same copy semantics - composing the same
    shared `_OperationRuleRegistry` mechanics. It differs only in what a rule
    returns (:class:`PulseDefinition` instead of a matrix) and in how a selected
    rule's failures are reported (see `_invoke_pulse_rule` and
    `PulseImplementationError`).

    A rule has the signature
    ``rule(operation, *, targets, model, calibration) -> PulseDefinition``.
    ``targets`` are ordered model-minted subsystem resource handles, not
    program register references or engine indices. A rule may inspect the
    immutable model and calibration but should return only reusable physical
    realization data.

    Use :func:`~fatqat.backends.default_superconducting_pulse_implementation_map`
    as the starting point when replacing one built-in realization. A
    :class:`~fatqat.backends.PulseBackend` copies the map passed to its
    constructor, so later registration changes do not alter that backend.

    Examples:
        Replace one unconstrained implementation while retaining the other
        defaults::

            implementations = default_superconducting_pulse_implementation_map()
            implementations.add(ops.CZ, custom_cz)
    """

    def __init__(self) -> None:
        self._registry: _OperationRuleRegistry[_PulseImplementation] = (
            _OperationRuleRegistry()
        )

    def add(
        self,
        op: Operation | type[Operation],
        implementation: "_PulseImplementation | Callable",
        *,
        device_operands: DeviceOperands | None = None,
    ) -> None:
        """Add an unconstrained or device-specific pulse implementation.

        Args:
            op: An :py:class:`~fatqat.operations.Operation` instance (e.g.
                `ops.CZ`) or subclass. Normalized to the operation's class for
                the registry key.
            implementation: A callable `f(operation, *, targets, model,
                calibration) -> PulseDefinition`, or an already-built
                `_PulseImplementation`.
            device_operands: An ordered hashable tuple identifying the
                device-level target this rule is restricted to. Omit for an
                unconstrained rule that applies to every legal target of the
                operation's arity.

        Raises:
            TypeError: If `op` is neither an :py:class:`~fatqat.operations.Operation`
                instance nor subclass, if its operation class has variable
                arity, or if `implementation` is not callable.
            ValueError: If `device_operands`' length does not match the
                operation's arity, or if `op` already has a registration in
                the other mode; see `MatrixImplementationMap.add` for why the two
                modes are mutually exclusive.
        """
        if device_operands is not None:
            self._registry.add_device_operands(
                op,
                device_operands,
                lambda op_cls: _wrap_pulse_rule(op_cls, implementation),
            )
            return
        self._registry.add_unconstrained(
            op, lambda op_cls: _wrap_pulse_rule(op_cls, implementation)
        )

    def supports(
        self,
        op: Operation | type[Operation],
        *,
        device_operands: DeviceOperands | None = None,
    ) -> bool:
        """Return whether this map has any rule for the operation family.

        Same semantics as `MatrixImplementationMap.supports`.
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
    ) -> _PulseImplementation | None:
        """Return the pulse implementation selected for an operation.

        Same lookup semantics as `MatrixImplementationMap.implementation_for`: with
        `device_operands` omitted, only the unconstrained rule is consulted;
        with `device_operands` given, a family with device-specific rules
        consults only those, while a family with only an unconstrained rule
        returns it for every device operands.
        """
        return self._registry.get(op, device_operands=device_operands)

    def supported_operations(self) -> frozenset[type[Operation]]:
        """Return every operation family with at least one implementation."""
        return self._registry.supported_operations()

    def device_operands_for(
        self, op: Operation | type[Operation]
    ) -> frozenset[DeviceOperands]:
        """Return the finite set of device operands selected for an operation.

        Same semantics as `MatrixImplementationMap.device_operands_for`.
        """
        return self._registry.device_operands_for(op)

    def remove(self, op: Operation | type[Operation]) -> None:
        """Remove a registered pulse implementation, if present.

        Removing an operation that was never registered is a no-op.
        """
        self._registry.remove(op)

    def copy(self) -> "PulseImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects are shared (not deep-copied) between the original and
        the copy, matching `MatrixImplementationMap.copy`. Later mutations of
        either map's registrations never affect the other.
        """
        clone = PulseImplementationMap()
        clone._registry = self._registry.copy()
        return clone
