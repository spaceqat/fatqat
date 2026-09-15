"""Export an optimized phase-modulated pulse to FATQAT pulse values.

The optimizer works in the dimensionless units of the paper, while FATQAT
emulates physical units (rad/us and us) through the three-level atom family.
This module converts a :class:``~.problem.CZPulse`` into the pulse values that
the family's :class:``~.backend.Atom3LevelEmulator`` executes.

Phase convention
----------------
Paper Eq. (19) writes the block Hamiltonian as
``H_q = (Omega_max / 2) sqrt(m_q) (cos(phi) sigma_x - sin(phi) sigma_y)``, and
the FATQAT adapter assembles
``H = (Omega sigma_+ + Omega^* sigma_-) / 2`` from the complex waveform
coefficient ``Omega``.  The two agree for

    Omega(t) = Omega_max * exp(-1j * phi(t)),

which is also the convention of the built-in CZ recipe of the family
(``coefficients = omega_1r * np.exp(-1j * phase)``).  :func:``cz_waveform``
applies that mapping, so a pulse optimized here can be run and verified with
the emulator unchanged.

The implemented phase gate has ``xi_01 = xi_10 = theta``, so
:func:``cz_pulse_definition`` appends one virtual Z rotation by ``-theta`` per
site.  With that correction the exported pulse realizes a strict CZ gate.
"""

from __future__ import annotations

import math

import numpy as np

from ....errors import BackendValidationError
from ...._pulse_values import PulseControl
from ...._waveforms import SampledWaveform
from ....operations import CZ
from ..._core.pulse import PhaseShift, PulseDefinition, PulseImplementationMap
from ..calibration import Atom3LevelCalibration, default_atom_3level_calibration
from ..model import Atom3LevelModel
from ..realization import _default_atom_3level_gate_implementation_map
from .problem import CZPulse, _require_positive_omega_max

#: Default number of waveform samples.  It matches the sample count of the
#: built-in CZ recipe of the family, so an exported pulse and the built-in
#: pulse have the same numerical resolution.
DEFAULT_SAMPLE_COUNT = 801


def cz_waveform(
    pulse: CZPulse,
    *,
    omega_max: float,
    sample_count: int | None = None,
) -> SampledWaveform:
    """Return the FATQAT waveform of an optimized CZ pulse.

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        omega_max: Peak Rabi frequency in rad/us; the physical pulse duration
            is ``pulse.duration / omega_max`` microseconds.
        sample_count: Number of waveform samples; ``None`` uses
            :data:``DEFAULT_SAMPLE_COUNT``.  A piecewise-constant pulse is
            smoothed onto the uniform sample grid, so a large count keeps the
            staircase, while a small count with a PMP pulse keeps the control
            cheap.

    Returns:
        A complex waveform with coefficients
        ``Omega(t) = omega_max exp(-1j phi(t))``, sampled on a uniform grid
        from ``0`` to the physical pulse duration.

    Raises:
        BackendValidationError: If ``omega_max`` or ``sample_count`` is
            invalid.
    """
    _require_positive_omega_max(omega_max)
    count = DEFAULT_SAMPLE_COUNT if sample_count is None else sample_count
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise BackendValidationError("sample_count must be an integer >= 2")
    sampled = pulse.resampled(count)
    times = np.linspace(0.0, pulse.duration, count) / omega_max
    coefficients = omega_max * np.exp(-1j * np.asarray(sampled.phases))
    return SampledWaveform(tuple(float(value) for value in times), tuple(coefficients))


def cz_pulse_definition(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    sample_count: int | None = None,
) -> PulseDefinition:
    """Return the pulse definition of an optimized global CZ gate.

    Args:
        model: Model whose Rydberg control and frame addresses are used.
        pulse: Pulse in the dimensionless units of the paper.
        omega_max: Peak Rabi frequency in rad/us.
        sample_count: Number of waveform samples; see :func:``cz_waveform``.

    Returns:
        A pulse definition that drives the Rydberg transition of both sites
        with the same waveform and shifts each site frame by ``-theta``, so
        that the resulting gate is a strict CZ gate.

    Raises:
        BackendValidationError: If ``model`` has the wrong type.
    """
    if not isinstance(model, Atom3LevelModel):
        raise BackendValidationError("model must be an Atom3LevelModel")
    waveform = cz_waveform(pulse, omega_max=omega_max, sample_count=sample_count)
    duration = pulse.duration / omega_max
    controls = tuple(
        PulseControl(model.control.rydberg(site), waveform) for site in (0, 1)
    )
    correction = float(np.mod(-pulse.theta, 2.0 * math.pi))
    actions = tuple(PhaseShift(model.frame(site), correction) for site in (0, 1))
    return PulseDefinition(duration, controls, actions)


def cz_gate_implementation_map(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    sample_count: int | None = None,
    calibration: Atom3LevelCalibration | None = None,
) -> PulseImplementationMap:
    """Return the standard atom gate map with the optimized CZ rule.

    The returned map implements ``RX``, ``RY`` and ``RZ`` with the family's
    default calibration and overrides ``CZ`` with the optimized pulse, so a
    program can prepare states with Raman rotations and then apply the
    optimized gate.

    Args:
        model: Model whose channel and frame addresses the rules use.
        pulse: Pulse in the dimensionless units of the paper.
        omega_max: Peak Rabi frequency in rad/us.
        sample_count: Number of waveform samples; see :func:``cz_waveform``.
        calibration: Gate recipes for the single-qubit rules; ``None`` uses
            :func:``~..calibration.default_atom_3level_calibration``.

    Returns:
        A pulse implementation map combining the default single-qubit rules
        with the optimized CZ rule.

    Raises:
        BackendValidationError: If the model or the calibration has the wrong
            type.
    """
    recipes = default_atom_3level_calibration() if calibration is None else calibration
    implementations = _default_atom_3level_gate_implementation_map(
        model=model, calibration=recipes
    )
    implementations.remove(CZ)
    implementations.add(
        CZ,
        lambda _operation, *, device_operands: _cz_definition(
            model,
            pulse,
            omega_max=omega_max,
            sample_count=sample_count,
            device_operands=device_operands,
        ),
    )
    return implementations


def _cz_definition(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    sample_count: int | None,
    device_operands: tuple[object, ...],
) -> PulseDefinition:
    """Return the CZ pulse definition for the sites of one gate."""
    if len(device_operands) != 2:
        raise BackendValidationError(
            "atom operation requires exactly 2 non-negative site ordinals"
        )
    sites = tuple(_site(operand) for operand in device_operands)
    waveform = cz_waveform(pulse, omega_max=omega_max, sample_count=sample_count)
    duration = pulse.duration / omega_max
    controls = tuple(
        PulseControl(model.control.rydberg(site), waveform) for site in sites
    )
    correction = float(np.mod(-pulse.theta, 2.0 * math.pi))
    actions = tuple(PhaseShift(model.frame(site), correction) for site in sites)
    return PulseDefinition(duration, controls, actions)


def _site(operand: object) -> int:
    """Return one validated site ordinal of a CZ operand pair."""
    if not isinstance(operand, int) or isinstance(operand, bool) or operand < 0:
        raise BackendValidationError(
            "atom operation requires exactly 2 non-negative site ordinals"
        )
    return operand
