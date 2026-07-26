"""Validated, engine-neutral resolved pulse values and SC native realization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Any

import numpy as np

from ...errors import BackendValidationError, UnsupportedOperationError
from ...operations.base import Operation
from ...operations.fixed_gates import CZGate, iSwapGate
from ...operations.parametric_gates import RX, RY, RZ
from .superconducting import (
    CalibrationSpec,
    ControlChannelRef,
    CouplingRef,
    FrameRef,
    PhysicsModel,
    SubsystemResourceRef,
)

_WAVEFORM_SAMPLES = 129


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendValidationError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value) or (nonnegative and value < 0):
        raise BackendValidationError(f"{name} must be finite and non-negative")
    return value


def _freeze(values: Any, *, dtype: type = complex) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SampledControl:
    """One sampled physical control with a local, independently timed grid."""

    channel: ControlChannelRef
    tlist: np.ndarray
    coefficients: np.ndarray
    start_offset_ns: float = 0.0

    def __post_init__(self) -> None:
        start_offset = _finite(
            self.start_offset_ns, "control start_offset_ns", nonnegative=True
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
        object.__setattr__(self, "start_offset_ns", start_offset)
        object.__setattr__(self, "tlist", _freeze(tlist, dtype=float))
        object.__setattr__(self, "coefficients", _freeze(coefficients))

    @property
    def duration_ns(self) -> float:
        return float(self.tlist[-1])


@dataclass(frozen=True)
class PhaseShift:
    """A post-block virtual-frame angle update with no physical duration."""

    frame: FrameRef
    angle_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angle_rad", _finite(self.angle_rad, "phase-shift angle_rad")
        )


@dataclass(frozen=True)
class PhaseSwap:
    """A post-block exchange of two virtual-drive frame ledgers."""

    first: FrameRef
    second: FrameRef

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise BackendValidationError(
                "phase swap requires two distinct frame references"
            )


FrameAction = PhaseShift | PhaseSwap
ResourceClaim = SubsystemResourceRef | CouplingRef


@dataclass(frozen=True)
class PulseBlock:
    """One atomic, unplaced or explicitly placed model-owned pulse block."""

    model: PhysicsModel
    duration_ns: float
    children: tuple[SampledControl, ...]
    resource_claims: tuple[ResourceClaim, ...]
    post_actions: tuple[FrameAction, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None
    start_ns: float | None = None

    def __post_init__(self) -> None:
        duration = _finite(
            self.duration_ns, "pulse-block duration_ns", nonnegative=True
        )
        if not self.resource_claims:
            raise BackendValidationError(
                "pulse block must claim at least one model resource"
            )
        if duration == 0.0 and self.children:
            raise BackendValidationError(
                "a zero-duration pulse block cannot contain physical controls"
            )
        if duration > 0.0 and not self.children:
            raise BackendValidationError(
                "a positive-duration pulse block requires physical controls"
            )
        seen_channels: set[ControlChannelRef] = set()
        required_claim_sets: list[set[ResourceClaim]] = []
        for child in self.children:
            if not isinstance(child, SampledControl):
                raise BackendValidationError(
                    "pulse-block children must be SampledControl values"
                )
            if not isinstance(child.channel, ControlChannelRef):
                raise BackendValidationError(
                    "pulse control has an unknown channel reference"
                )
            control_ordinal = self.model.bind_control(child.channel)
            if child.channel.kind == "exchange":
                coupling = self.model.couplings[control_ordinal]
                required_claim_sets.append(
                    {
                        self.model.resource(subsystem_id)
                        for subsystem_id in coupling.subsystem_ids
                    }
                )
            else:
                required_claim_sets.append(
                    {self.model.resource(self.model.subsystem_ids[control_ordinal])}
                )
            if child.channel in seen_channels:
                raise BackendValidationError(
                    "pulse block cannot implicitly sum controls on one channel"
                )
            seen_channels.add(child.channel)
            if child.start_offset_ns + child.duration_ns > duration + 1e-12:
                raise BackendValidationError(
                    "control extends beyond its enclosing pulse block"
                )
        seen_resources: set[ResourceClaim] = set()
        for resource in self.resource_claims:
            if isinstance(resource, SubsystemResourceRef):
                self.model.bind_resource(resource)
            elif isinstance(resource, CouplingRef):
                self.model.bind_coupling(resource)
            else:
                raise BackendValidationError(
                    "pulse block has an unknown resource claim"
                )
            if resource in seen_resources:
                raise BackendValidationError(
                    "pulse block has a duplicate resource claim"
                )
            seen_resources.add(resource)
        for required_claims in required_claim_sets:
            if not required_claims <= seen_resources:
                raise BackendValidationError(
                    "pulse block resource claims do not cover a driven control"
                )
        for action in self.post_actions:
            if isinstance(action, PhaseShift):
                self.model.bind_frame(action.frame)
            elif isinstance(action, PhaseSwap):
                self.model.bind_frame(action.first)
                self.model.bind_frame(action.second)
            else:
                raise BackendValidationError("pulse block has an unknown frame action")
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
        if self.start_ns is not None:
            object.__setattr__(
                self,
                "start_ns",
                _finite(self.start_ns, "pulse-block start_ns", nonnegative=True),
            )
        object.__setattr__(self, "duration_ns", duration)
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "resource_claims", tuple(self.resource_claims))
        object.__setattr__(self, "post_actions", tuple(self.post_actions))


def _sample_grid(duration_ns: float) -> np.ndarray:
    return np.linspace(0.0, duration_ns, _WAVEFORM_SAMPLES)


def _hann(tlist: np.ndarray, duration_ns: float, peak: float) -> np.ndarray:
    return peak * np.sin(pi * tlist / duration_ns) ** 2


def _cumulative_trapezoid(values: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(values, dtype=float)
    phase[1:] = np.cumsum((values[1:] + values[:-1]) * np.diff(tlist) / 2.0)
    return phase


def _single_resource_claims(
    model: PhysicsModel, subsystem_id: str
) -> tuple[ResourceClaim, ...]:
    return (model.resource(subsystem_id),)


def _pair_resource_claims(
    model: PhysicsModel, first: str, second: str
) -> tuple[ResourceClaim, ...]:
    return (
        model.resource(first),
        model.resource(second),
        model.coupling(first, second),
    )


def _target_ids(
    model: PhysicsModel, targets: tuple[SubsystemResourceRef, ...], expected: int
) -> tuple[str, ...]:
    if len(targets) != expected:
        raise BackendValidationError(
            f"native operation requires {expected} model resource target(s)"
        )
    ordinals = tuple(model.bind_resource(target) for target in targets)
    if len(set(ordinals)) != expected:
        raise BackendValidationError(
            "native operation requires distinct model resource targets"
        )
    return tuple(model.subsystem_ids[ordinal] for ordinal in ordinals)


def realize_native_operation(
    operation: Operation,
    targets: tuple[SubsystemResourceRef, ...],
    *,
    model: PhysicsModel,
    calibration: CalibrationSpec,
    condition: tuple[tuple[int, int], ...] | None = None,
) -> PulseBlock:
    """Realize one supported operation as one atomic unplaced ``PulseBlock``."""
    if calibration.key != model.key:
        raise BackendValidationError("calibration does not match the pulse model")
    if isinstance(operation, (RX, RY)):
        (subsystem_id,) = _target_ids(model, targets, 1)
        recipe = calibration.recipe("rx_ry")
        duration = float(recipe["duration_ns"])
        drag_coefficient = float(recipe["drag_coefficient"])
        tlist = _sample_grid(duration)
        # The durable anharmonicity is in GHz. Controls use angular inverse ns,
        # so convert it at this realization boundary before applying DRAG.
        alpha = (
            2 * pi * model.subsystems[model.bind_resource(targets[0])].anharmonicity_ghz
        )
        theta = _finite(operation.theta, "rotation angle")
        p = _hann(tlist, duration, theta / duration)
        dp = theta * pi * np.sin(2 * pi * tlist / duration) / duration**2
        x0 = p - p**3 / alpha**2
        zv = -2 * p**2 / alpha
        y0 = -drag_coefficient * dp / (alpha + zv)
        phase = _cumulative_trapezoid(zv, tlist)
        envelope = (x0 + 1j * y0) * np.exp(1j * phase)
        if isinstance(operation, RY):
            envelope *= 1j
        return PulseBlock(
            model=model,
            duration_ns=duration,
            children=(
                SampledControl(model.drive_control(subsystem_id), tlist, envelope),
            ),
            resource_claims=_single_resource_claims(model, subsystem_id),
            post_actions=(PhaseShift(model.frame(subsystem_id), float(phase[-1])),),
            condition=condition,
        )
    if isinstance(operation, RZ):
        (subsystem_id,) = _target_ids(model, targets, 1)
        angle = _finite(operation.theta, "rotation angle")
        return PulseBlock(
            model=model,
            duration_ns=0.0,
            children=(),
            resource_claims=_single_resource_claims(model, subsystem_id),
            post_actions=(PhaseShift(model.frame(subsystem_id), angle),),
            condition=condition,
        )
    if isinstance(operation, iSwapGate):
        first, second = _target_ids(model, targets, 2)
        duration = float(calibration.recipe("iswap")["duration_ns"])
        tlist = _sample_grid(duration)
        # With the adapter's ``J(a†b + ab†)`` Hamiltonian, a negative signed
        # area produces fatqat's public ``+i`` iSWAP convention.
        exchange = _hann(tlist, duration, -pi / duration)
        return PulseBlock(
            model=model,
            duration_ns=duration,
            children=(
                SampledControl(model.exchange_control(first, second), tlist, exchange),
            ),
            resource_claims=_pair_resource_claims(model, first, second),
            post_actions=(PhaseSwap(model.frame(first), model.frame(second)),),
            condition=condition,
        )
    if isinstance(operation, CZGate):
        first, second = _target_ids(model, targets, 2)
        edge_recipe = _cz_recipe(calibration, first, second)
        if edge_recipe["detuning_subsystem"] != first:
            raise BackendValidationError(
                "CZ target order does not match the declared detuning orientation"
            )
        duration = float(edge_recipe["duration_ns"])
        ramp = float(edge_recipe["ramp_duration_ns"])
        parked_duration = duration - 2 * ramp
        detuning_grid = _sample_grid(duration)
        ramp_shape = np.ones_like(detuning_grid)
        rising = detuning_grid < ramp
        falling = detuning_grid > duration - ramp
        ramp_shape[rising] = (1 - np.cos(pi * detuning_grid[rising] / ramp)) / 2
        ramp_shape[falling] = (
            1 - np.cos(pi * (duration - detuning_grid[falling]) / ramp)
        ) / 2
        detuning = 2 * pi * float(edge_recipe["detuning_ghz"]) * ramp_shape
        exchange_grid = _sample_grid(parked_duration)
        exchange = _hann(exchange_grid, parked_duration, sqrt(2) * pi / parked_duration)
        corrections = edge_recipe["phase_corrections_rad"]
        return PulseBlock(
            model=model,
            duration_ns=duration,
            children=(
                SampledControl(model.detuning_control(first), detuning_grid, detuning),
                SampledControl(
                    model.exchange_control(first, second), exchange_grid, exchange, ramp
                ),
            ),
            resource_claims=_pair_resource_claims(model, first, second),
            post_actions=tuple(
                PhaseShift(model.frame(subsystem_id), float(corrections[subsystem_id]))
                for subsystem_id in (first, second)
            ),
            condition=condition,
        )
    raise UnsupportedOperationError(
        f"{type(operation).__name__} is not supported by the SC pulse model"
    )


def _cz_recipe(calibration: CalibrationSpec, first: str, second: str) -> Any:
    for edge in calibration.recipe("cz")["edges"]:
        if frozenset(edge["subsystems"]) == frozenset((first, second)):
            return edge
    raise BackendValidationError(
        f"calibration has no CZ recipe for declared edge {first!r}-{second!r}"
    )
