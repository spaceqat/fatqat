"""Validated pulse-plan values on their owning model's time axis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from ..errors import BackendValidationError
from .lindblad import ResolvedLindbladTerm
from .superconducting import (
    ControlChannelRef,
    CouplingRef,
    FrameRef,
    PhysicsModel,
    SubsystemResourceRef,
)


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
        if not self.resource_claims:
            raise BackendValidationError(
                "pulse block must claim at least one model resource"
            )
        if duration == 0.0 and self.controls:
            raise BackendValidationError(
                "a zero-duration pulse block cannot contain physical controls"
            )
        if duration > 0.0 and not self.controls:
            raise BackendValidationError(
                "a positive-duration pulse block requires physical controls"
            )
        seen_channels: set[ControlChannelRef] = set()
        required_claim_sets: list[set[ResourceClaim]] = []
        for child in self.controls:
            if not isinstance(child, SampledControl):
                raise BackendValidationError(
                    "pulse-block controls must be SampledControl values"
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
                    | {self.model.coupling(*coupling.subsystem_ids)}
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
            if child.start_offset + child.duration > duration + 1e-12:
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
