"""Calibrated superconducting-operation realization into pulse-plan blocks."""

from __future__ import annotations

from math import pi, sqrt
from typing import Any

import numpy as np

from ..errors import BackendValidationError, UnsupportedOperationError
from ..operations.base import Operation
from ..operations.fixed_gates import CZGate, iSwapGate
from ..operations.parametric_gates import RX, RY, RZ
from .pulse import PhaseShift, PhaseSwap, PulseBlock, ResourceClaim, SampledControl
from .superconducting import CalibrationSpec, PhysicsModel, SubsystemResourceRef

_WAVEFORM_SAMPLES = 129


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendValidationError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise BackendValidationError(f"{name} must be finite and non-negative")
    return value


def _sample_grid(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, _WAVEFORM_SAMPLES)


def _hann(tlist: np.ndarray, duration: float, peak: float) -> np.ndarray:
    return peak * np.sin(pi * tlist / duration) ** 2


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


def realize_calibrated_operation(
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
            duration=duration,
            controls=(
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
            duration=0.0,
            controls=(),
            resource_claims=_single_resource_claims(model, subsystem_id),
            post_actions=(PhaseShift(model.frame(subsystem_id), angle),),
            condition=condition,
        )
    if isinstance(operation, iSwapGate):
        first, second = _target_ids(model, targets, 2)
        duration = float(calibration.recipe("iswap")["duration_ns"])
        tlist = _sample_grid(duration)
        # The signed exchange area selects fatqat's public +i iSWAP convention.
        exchange = _hann(tlist, duration, -pi / duration)
        return PulseBlock(
            model=model,
            duration=duration,
            controls=(
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
            duration=duration,
            controls=(
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
