"""Upstream converters from calibration data to channel descriptors.

The matrix family's channel catalog only ever consumes already-resolved rate
parameters - operations carry no duration, so the library never derives
time-dependent rates itself. This module is the explicit upstream step: the
caller supplies the device timescales (``t1``, ``t2``) *and* how long the
noisy operation takes, and gets back plain descriptors to register on a
`NoiseModel`. Backends generating default noise from a calibration profile
call the same converters (see
``FakeSuperconducting4x4Backend.default_noise_model``).
"""

from __future__ import annotations

from math import exp

from .catalog import AmplitudeDamping, PhaseDamping


def relaxation_channels(
    t1: float, t2: float, duration: float
) -> tuple[AmplitudeDamping, PhaseDamping]:
    """Convert qubit ``T1``/``T2`` timescales into relaxation channels.

    Returns the pair of single-qubit channels that reproduces thermal
    relaxation over ``duration``: populations decay toward the ground state
    at rate ``gamma = 1 - exp(-duration/t1)``, and coherences decay by the
    total factor ``exp(-duration/t2)``. Amplitude damping alone already
    shrinks coherences by ``exp(-duration/(2*t1))``, so the dephasing channel
    carries only the residual, which is why ``t2 <= 2*t1`` is required (the
    physical bound: pure dephasing cannot be negative).

    Attach both returned channels to the same single-qubit gate occurrence
    (order does not matter; the two commute).

    Args:
        t1: Energy-relaxation timescale, in the same time unit as
            ``duration``. Must be positive.
        t2: Total dephasing timescale. Must satisfy ``0 < t2 <= 2 * t1``.
        duration: How long the noisy operation takes. Must be >= 0; zero
            yields identity channels.

    Returns:
        ``(AmplitudeDamping, PhaseDamping)`` descriptors for one qubit.

    Raises:
        ValueError: If a timescale or the duration violates the bounds above.
    """
    if not t1 > 0:
        raise ValueError(f"t1 must be positive, got {t1!r}")
    if not 0 < t2 <= 2 * t1:
        raise ValueError(f"t2 must satisfy 0 < t2 <= 2*t1, got t2={t2!r}, t1={t1!r}")
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration!r}")
    gamma = 1.0 - exp(-duration / t1)
    # Residual pure dephasing on top of the damping-induced part: the
    # catalog's PhaseDamping(p) leaves qubit coherence at factor (1 - p).
    dephasing_rate = 1.0 / t2 - 1.0 / (2.0 * t1)
    p = 1.0 - exp(-duration * dephasing_rate)
    return AmplitudeDamping(gammas=(gamma,)), PhaseDamping(p=p)
