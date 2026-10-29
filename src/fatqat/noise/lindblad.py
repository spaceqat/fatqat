"""Lindblad-operator implementations shared by pulse emulators."""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np

from ..errors import BackendValidationError, UnsupportedOperationError
from ..implementation.matrices import clock_matrix, shift_matrix
from ..implementation.base_map import _BaseImplementationMap
from .base import Channel
from .catalog import AmplitudeDamping, Depolarizing, PhaseDamping
from .relaxation import ThermalRelaxation
from .transition_relaxation import TransitionRelaxation, transition_operator


def _resolve_channel_type(channel_type: type[Channel]) -> type[Channel]:
    if not isinstance(channel_type, type) or not issubclass(channel_type, Channel):
        raise TypeError(f"expected a Channel subclass, got {channel_type!r}")
    return channel_type


class LindbladImplementationMap(_BaseImplementationMap[Callable]):
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
    Registration and lookup accept channel classes, not instances.
    """

    def __init__(self) -> None:
        super().__init__(_resolve_channel_type)

    def add(self, op: type[Channel], implementation: Callable) -> None:
        """Register or replace a callable for an exact channel type."""
        if not callable(implementation):
            raise TypeError(f"rule must be callable, got {implementation!r}")
        self._add_unconstrained(op, lambda _: implementation)

    def supported_channels(self) -> frozenset[type[Channel]]:
        """Return an immutable snapshot of the registered channel types."""
        return self._registered_types()


def _qubit_amplitude_operator(
    rate: float,
    *,
    physical_dimension: int,
    label: str,
) -> np.ndarray:
    """Build the conventional qubit lowering collapse operator."""
    if physical_dimension != 2:
        raise BackendValidationError(
            f"{label} requires physical dimension 2, got {physical_dimension}"
        )
    operator = np.zeros((2, 2), dtype=complex)
    operator[0, 1] = sqrt(rate)
    return operator


def amplitude_damping_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve rate-form qubit amplitude damping to one collapse operator."""
    assert isinstance(channel, AmplitudeDamping)
    if channel.rate is None:
        raise BackendValidationError(
            "AmplitudeDamping requires authored rate mode on a pulse backend"
        )
    return (
        _qubit_amplitude_operator(
            channel.rate,
            physical_dimension=physical_dimension,
            label="AmplitudeDamping",
        ),
    )


def transition_relaxation_lindblad_rule(
    channel: Channel, *, physical_dimension: int
) -> tuple[np.ndarray, ...]:
    """Resolve one authored transition jump to one collapse operator."""
    assert isinstance(channel, TransitionRelaxation)
    if channel.rate is None:
        raise BackendValidationError(
            "TransitionRelaxation requires authored rate mode on a pulse backend"
        )
    return (sqrt(channel.rate) * transition_operator(channel, physical_dimension),)


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
    """Resolve qubit T1/T2 relaxation into its two physical contributions."""
    assert isinstance(channel, ThermalRelaxation)
    amplitude = _qubit_amplitude_operator(
        channel.amplitude_rate,
        physical_dimension=physical_dimension,
        label="ThermalRelaxation",
    )
    if channel.pure_dephasing_rate == 0.0:
        return (amplitude,)
    number = np.diag(np.arange(2, dtype=float)).astype(complex)
    return amplitude, sqrt(2 * channel.pure_dephasing_rate) * number


def resolve_lindblad_operators(
    channel: Channel,
    *,
    implementation_map: LindbladImplementationMap,
    physical_dimension: int,
) -> tuple[np.ndarray, ...]:
    """Resolve a channel to validated local Lindblad-operator matrices."""
    rule = implementation_map.implementation_for(type(channel))
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
