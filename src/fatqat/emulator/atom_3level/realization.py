"""Compile standard three-level-atom gates into portable pulse maps."""

from __future__ import annotations

from math import isfinite, pi

import numpy as np

from ...errors import BackendValidationError
from ..._pulse_values import PulseControl
from ...operations import CZ, RX, RY, RZ, Operation
from ...waveforms import SampledWaveform
from .._core.pulse import PhaseShift, PulseDefinition, PulseImplementationMap
from .calibration import Atom3LevelCalibration
from .model import Atom3LevelModel

_CZ_SAMPLE_POINT_COUNT = 801


def _ordinals(device_operands: tuple[object, ...], expected: int) -> tuple[int, ...]:
    if len(device_operands) != expected or any(
        type(ordinal) is not int or ordinal < 0 for ordinal in device_operands
    ):
        raise BackendValidationError(
            f"atom operation requires exactly {expected} non-negative site ordinals"
        )
    resolved = tuple(device_operands)
    if len(set(resolved)) != len(resolved):
        raise BackendValidationError("atom operation targets must be distinct")
    return resolved


def _theta(operation: Operation) -> float:
    value = operation.theta
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise BackendValidationError("atom rotation angle must be finite")
    return float(value)


def default_atom_3level_gate_implementation_map(
    *, model: Atom3LevelModel, calibration: Atom3LevelCalibration
) -> PulseImplementationMap:
    """Compile a fresh arrangement-independent standard atom pulse map.

    ``model`` is a required source-model seam for later pulse-design work. The
    first version intentionally does not inspect or retain it: C6 and geometry
    remain target-evolution facts rather than inputs to the fixed pulse recipe.
    """
    if not isinstance(model, Atom3LevelModel):
        raise BackendValidationError("model must be an Atom3LevelModel")
    if not isinstance(calibration, Atom3LevelCalibration):
        raise BackendValidationError("calibration must be an Atom3LevelCalibration")

    omega_01 = calibration.omega_01_angular_per_us
    omega_1r = calibration.omega_1r_angular_per_us
    cz_duration = calibration.cz_duration_us
    phase_amplitude = calibration.phase_amplitude_rad
    phase_rate = calibration.cz_phase_rate_angular_per_us
    phase_offset = calibration.phase_offset_rad
    linear_phase_rate = calibration.cz_linear_phase_rate_angular_per_us
    local_z_correction = calibration.local_z_correction_rad

    def raman(operation: RX | RY, *, device_operands: tuple[object, ...]):
        (target,) = _ordinals(device_operands, 1)
        theta = _theta(operation)
        if theta == 0.0:
            return PulseDefinition(0.0, ())
        phase = 0.0 if isinstance(operation, RX) else -pi / 2
        if theta < 0:
            phase += pi
        duration = abs(theta) / omega_01
        coefficient = omega_01 * np.exp(-1j * phase)
        return PulseDefinition(
            duration,
            (
                PulseControl(
                    model.control.raman(target),
                    SampledWaveform((0.0, duration), (coefficient, coefficient)),
                ),
            ),
        )

    def rz(operation: RZ, *, device_operands: tuple[object, ...]):
        (target,) = _ordinals(device_operands, 1)
        return PulseDefinition(
            0.0,
            (),
            (PhaseShift(model.frame(target), _theta(operation)),),
        )

    def cz(_operation: CZ, *, device_operands: tuple[object, ...]):
        targets = _ordinals(device_operands, 2)
        grid = np.linspace(0.0, cz_duration, _CZ_SAMPLE_POINT_COUNT)
        phase = (
            phase_amplitude * np.cos(phase_rate * grid - phase_offset)
            + linear_phase_rate * grid
        )
        coefficients = omega_1r * np.exp(-1j * phase)
        return PulseDefinition(
            cz_duration,
            tuple(
                PulseControl(
                    model.control.rydberg(target),
                    SampledWaveform(grid, coefficients),
                )
                for target in targets
            ),
            tuple(
                PhaseShift(model.frame(target), local_z_correction)
                for target in targets
            ),
        )

    implementations = PulseImplementationMap()
    implementations.add(RX, raman)
    implementations.add(RY, raman)
    implementations.add(RZ, rz)
    implementations.add(CZ, cz)
    return implementations
