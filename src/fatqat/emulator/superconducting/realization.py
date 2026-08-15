"""Compile standard transmon gate implementations into portable pulse maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from ... import operations as ops
from ..._pulse_values import ControlChannel, PulseControl
from ...errors import BackendValidationError
from ...operations.base import Operation
from ...operations.parametric_gates import RY
from ...waveforms import SampledWaveform
from .._core.target import Frame
from .._core.pulse import PhaseShift, PhaseSwap, PulseDefinition, PulseImplementationMap
from .._core.value_validation import _finite
from .calibration import TransmonCalibration
from .model import TransmonModel, angular_rate_from_ghz

_WAVEFORM_SAMPLES = 129


@dataclass(frozen=True)
class _DragContext:
    alpha: float
    drive: ControlChannel
    frame: Frame
    duration: float
    coefficient: float


def _sample_grid(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, _WAVEFORM_SAMPLES)


def _hann(tlist: np.ndarray, duration: float, peak: float) -> np.ndarray:
    return peak * np.sin(pi * tlist / duration) ** 2


def _cumulative_trapezoid(values: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(values, dtype=float)
    phase[1:] = np.cumsum((values[1:] + values[:-1]) * np.diff(tlist) / 2.0)
    return phase


def _drag_definition(operation: Operation, context: _DragContext) -> PulseDefinition:
    theta = _finite(operation.theta, "rotation angle")
    tlist = _sample_grid(context.duration)
    p = _hann(tlist, context.duration, theta / context.duration)
    dp = theta * pi * np.sin(2 * pi * tlist / context.duration) / context.duration**2
    x0 = p - p**3 / context.alpha**2
    zv = -2 * p**2 / context.alpha
    y0 = -context.coefficient * dp / (context.alpha + zv)
    phase = _cumulative_trapezoid(zv, tlist)
    envelope = (x0 + 1j * y0) * np.exp(1j * phase)
    if isinstance(operation, RY):
        envelope *= 1j
    return PulseDefinition(
        context.duration,
        (PulseControl(context.drive, SampledWaveform(tlist, envelope)),),
        (PhaseShift(context.frame, float(phase[-1])),),
    )


def _iswap_definition(
    model: TransmonModel,
    calibration: TransmonCalibration,
    first: str,
    second: str,
) -> PulseDefinition:
    duration = calibration._iswap_duration_ns
    tlist = _sample_grid(duration)
    exchange = _hann(tlist, duration, -pi / duration)
    return PulseDefinition(
        duration,
        (
            PulseControl(
                model.exchange_control(first, second),
                SampledWaveform(tlist, exchange),
            ),
        ),
        (PhaseSwap(model.frame(first), model.frame(second)),),
    )


def _cz_definition(
    model: TransmonModel,
    calibration: TransmonCalibration,
    first: str,
    second: str,
) -> PulseDefinition:
    duration = calibration._cz_duration_ns(first, second)
    ramp = calibration._cz_ramp_duration_ns(first, second)
    parked_duration = duration - 2 * ramp
    detuning_grid = _sample_grid(duration)
    ramp_shape = np.ones_like(detuning_grid)
    if ramp > 0:
        rising = detuning_grid < ramp
        falling = detuning_grid > duration - ramp
        ramp_shape[rising] = (1 - np.cos(pi * detuning_grid[rising] / ramp)) / 2
        ramp_shape[falling] = (
            1 - np.cos(pi * (duration - detuning_grid[falling]) / ramp)
        ) / 2
    detuning = (
        angular_rate_from_ghz(calibration._cz_detuning_ghz(first, second)) * ramp_shape
    )
    exchange_grid = _sample_grid(parked_duration)
    exchange = _hann(exchange_grid, parked_duration, sqrt(2) * pi / parked_duration)
    detuning_subsystem = calibration._cz_detuning_subsystem(first, second)
    detuning_phase = float(np.trapezoid(detuning, detuning_grid))
    return PulseDefinition(
        duration,
        (
            PulseControl(
                model.detuning_control(detuning_subsystem),
                SampledWaveform(detuning_grid, detuning),
            ),
            PulseControl(
                model.exchange_control(first, second),
                SampledWaveform(exchange_grid, exchange),
                ramp,
            ),
        ),
        (PhaseShift(model.frame(detuning_subsystem), detuning_phase),),
    )


def default_transmon_gate_implementation_map(
    *, model: TransmonModel, calibration: TransmonCalibration
) -> PulseImplementationMap:
    """Compile a fresh standard pulse map from source model and calibration."""
    if not isinstance(model, TransmonModel):
        raise BackendValidationError("model must be a TransmonModel")
    if not isinstance(calibration, TransmonCalibration):
        raise BackendValidationError("calibration must be a TransmonCalibration")

    drag_contexts = {
        subsystem.id: _DragContext(
            angular_rate_from_ghz(subsystem.anharmonicity_ghz),
            model.drive_control(subsystem.id),
            model.frame(subsystem.id),
            calibration._rx_ry_duration_ns,
            calibration._rx_ry_drag_coefficient,
        )
        for subsystem in model.subsystems
    }
    rz_frames = {
        subsystem.id: model.frame(subsystem.id) for subsystem in model.subsystems
    }

    def rx_ry(operation: Operation, *, device_operands: tuple[object, ...]):
        try:
            context = drag_contexts[device_operands[0]]
        except (IndexError, KeyError):
            raise BackendValidationError(
                f"compiled transmon map has no subsystem {device_operands!r}"
            ) from None
        return _drag_definition(operation, context)

    def rz(operation: Operation, *, device_operands: tuple[object, ...]):
        try:
            frame = rz_frames[device_operands[0]]
        except (IndexError, KeyError):
            raise BackendValidationError(
                f"compiled transmon map has no subsystem {device_operands!r}"
            ) from None
        return PulseDefinition(
            0.0,
            (),
            (PhaseShift(frame, _finite(operation.theta, "rotation angle")),),
        )

    implementations = PulseImplementationMap()
    implementations.add(ops.RX, rx_ry)
    implementations.add(ops.RY, rx_ry)
    implementations.add(ops.RZ, rz)
    for coupling in model.couplings:
        first, second = coupling.subsystem_ids
        for ordered in ((first, second), (second, first)):
            implementations.add(
                ops.iSwap,
                _iswap_definition(model, calibration, *ordered),
                device_operands=ordered,
            )
            implementations.add(
                ops.CZ,
                _cz_definition(model, calibration, *ordered),
                device_operands=ordered,
            )
    return implementations
