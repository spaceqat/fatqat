"""Lindblad-operator implementations shared by pulse emulators."""

from __future__ import annotations

from math import sqrt

import numpy as np

from ..errors import BackendValidationError, UnsupportedOperationError
from ..implementation.matrices import clock_matrix, shift_matrix
from .base import Channel, _ChannelImplementationRegistry
from .catalog import AmplitudeDamping, Depolarizing, PhaseDamping
from .relaxation import ThermalRelaxation


class LindbladImplementationMap(_ChannelImplementationRegistry):
    """Map exact channel types to local Lindblad rules.

    A registered rule has the call shape
    ``rule(channel, *, physical_dimension=dimension)`` and returns a nonempty
    tuple of local Lindblad-operator matrices. Every result must be a NumPy
    array of shape ``(dimension, dimension)``. Rules receive neither a target
    identity nor a duration: the channel supplies the rate or time
    parameters, and the pulse emulator controls how long the resolved
    Lindblad-operator noise acts.

    FATQAT validates only the returned count, array types, and shapes. A custom
    rule is responsible for the physical meaning and units of its operators.
    Lookup is by exact channel type.
    """


def _amplitude_lindblad_operator(
    rates: tuple[float, ...], physical_dimension: int
) -> np.ndarray:
    if len(rates) != physical_dimension - 1:
        raise BackendValidationError(
            f"AmplitudeDamping needs {physical_dimension - 1} rate value(s) for "
            f"dimension {physical_dimension}, got {len(rates)}"
        )
    operator = np.zeros((physical_dimension, physical_dimension), dtype=complex)
    for level, rate in enumerate(rates, start=1):
        operator[level - 1, level] = sqrt(rate)
    return operator


def amplitude_damping_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve amplitude damping to its local ladder-jump operator."""
    assert isinstance(channel, AmplitudeDamping)
    if channel.rate is None:
        raise BackendValidationError(
            "AmplitudeDamping requires authored rate mode on a pulse backend"
        )
    return (_amplitude_lindblad_operator(channel.rate, physical_dimension),)


def phase_damping_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve phase damping to its convention-matched local number operator."""
    assert isinstance(channel, PhaseDamping)
    rate = channel.rate
    if rate is None:
        raise BackendValidationError(
            "PhaseDamping requires authored rate or t_phi mode on a pulse backend"
        )
    number = np.diag(np.arange(physical_dimension, dtype=float)).astype(complex)
    return (sqrt(2 * rate) * number,)


def depolarizing_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve rate-form depolarization to nonidentity Weyl operators.

    The ``sqrt(rate) / d`` scaling makes the returned operators produce
    ``rate * (trace(rho) * I / d - rho)`` in every finite dimension.

    Args:
        channel: A depolarizing noise object in rate mode.
        physical_dimension: Local Hilbert-space dimension of the target site.

    Returns:
        The ``d**2 - 1`` scaled nonidentity Weyl jump operators.

    Raises:
        BackendValidationError: If the channel is in probability mode.
    """
    assert isinstance(channel, Depolarizing)
    if channel.rate is None:
        raise BackendValidationError(
            "Depolarizing requires authored rate mode on a pulse backend"
        )
    scale = sqrt(channel.rate) / physical_dimension
    return tuple(
        scale
        * (
            shift_matrix(physical_dimension, shift)
            @ clock_matrix(physical_dimension, phase)
        )
        for shift in range(physical_dimension)
        for phase in range(physical_dimension)
        if (shift, phase) != (0, 0)
    )


def thermal_relaxation_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve T1/T2 relaxation into amplitude and residual-dephasing operators."""
    assert isinstance(channel, ThermalRelaxation)
    amplitude = _amplitude_lindblad_operator(
        tuple(level * channel.amplitude_rate for level in range(1, physical_dimension)),
        physical_dimension,
    )
    if channel.pure_dephasing_rate == 0.0:
        return (amplitude,)
    number = np.diag(np.arange(physical_dimension, dtype=float)).astype(complex)
    return amplitude, sqrt(2 * channel.pure_dephasing_rate) * number


def default_lindblad_implementation_map() -> LindbladImplementationMap:
    """Return a fresh standard map for transmon Lindblad simulation.

    The map registers `AmplitudeDamping`, `PhaseDamping`, and
    `ThermalRelaxation`. These objects must use the rate/time forms
    accepted by the pulse backend. `Depolarizing` and `PauliChannel` are not
    registered by this factory.

    Each call returns an independent registration container. Individual
    emulator families may select a different built-in map.

    Returns:
        A new map containing the three standard rule registrations.
    """
    implementations = LindbladImplementationMap()
    implementations.add(AmplitudeDamping, amplitude_damping_lindblad_rule)
    implementations.add(PhaseDamping, phase_damping_lindblad_rule)
    implementations.add(ThermalRelaxation, thermal_relaxation_lindblad_rule)
    return implementations


def resolve_lindblad_operators(
    channel: Channel,
    *,
    implementation_map: LindbladImplementationMap,
    physical_dimension: int,
) -> tuple[np.ndarray, ...]:
    """Resolve a channel to validated local Lindblad-operator matrices."""
    rule = implementation_map.get(type(channel))
    if rule is None:
        raise UnsupportedOperationError(
            f"{type(channel).__name__} has no Lindblad implementation on this backend"
        )
    operators = tuple(rule(channel, physical_dimension=physical_dimension))
    if not operators:
        raise BackendValidationError(
            f"{type(channel).__name__} resolved to no Lindblad operators"
        )
    expected_shape = (physical_dimension, physical_dimension)
    for operator in operators:
        if not isinstance(operator, np.ndarray) or operator.shape != expected_shape:
            raise BackendValidationError(
                f"{type(channel).__name__} resolved to a Lindblad operator of "
                f"shape {getattr(operator, 'shape', type(operator))}, "
                f"expected {expected_shape}"
            )
    return operators
