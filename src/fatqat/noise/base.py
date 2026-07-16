"""Channel-noise abstraction: the descriptor marker, the rule protocol, the
rule registry, and the CPTP completeness check.

A channel descriptor (`Channel` subclass) carries physical parameters only,
never arrays - the same way an :py:class:`~fatqat.operations.Operation` like
``RX(0.3)`` never stores its matrix. A `ChannelImplementation` rule turns a
descriptor plus its resolution-time ``targets`` into a bare tuple of Kraus
arrays, mirroring how a matrix-implementation rule produces a bare matrix.
`ChannelImplementationMap` is the matrix family's registry from descriptor
type to rule; it exists (instead of reusing the gate implementation map)
because a channel resolves to a *tuple* of Kraus operators here but would
resolve to collapse operators in a future pulse family - the same descriptor
means different mathematical objects per backend family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..errors import BackendValidationError
from ..registers import RegisterRef


class Channel:
    """Base marker for channel descriptors.

    Concrete subclasses (see ``noise.catalog``) are frozen dataclasses holding
    physical parameters only - rates, probabilities - never Kraus arrays. The
    array computation belongs entirely to the `ChannelImplementation` rule
    registered for the subclass.
    """


class ChannelImplementation:
    """Base class for a channel implementation rule.

    A rule receives the bare `Channel` descriptor plus the ``targets``
    :py:class:`~fatqat.registers.RegisterRef` tuple by keyword and returns the
    channel's Kraus operators as a bare ``tuple[np.ndarray, ...]`` - no
    wrapper type, mirroring how a matrix-implementation rule returns a bare
    matrix. ``targets`` lets a rule read ``targets[0].register.dim`` to build
    dimension-dependent operators. Most rules are plain functions with this
    call shape; subclass only for a stateful or configured rule.
    """

    def __call__(
        self, channel: Channel, *, targets: tuple[RegisterRef, ...]
    ) -> tuple[np.ndarray, ...]:
        raise NotImplementedError


class ChannelImplementationMap:
    """Resolve channel descriptor types to their Kraus-producing rules.

    A dumb class-keyed registry: it never invokes rules itself and holds no
    layout or payload knowledge. The map is static per backend instance -
    which `NoiseModel` a run uses has no bearing on what the map can resolve.
    Its coverage is the backend's channel capability declaration: a descriptor
    type without a registered rule is unsupported.
    """

    def __init__(self) -> None:
        self._rules: dict[type[Channel], ChannelImplementation | Callable] = {}

    def register(
        self,
        channel_type: type[Channel],
        rule: ChannelImplementation | Callable,
    ) -> None:
        """Register the rule used to resolve one channel descriptor type.

        Args:
            channel_type: `Channel` subclass to key the registry by.
            rule: Callable with the `ChannelImplementation` call shape
                (``rule(channel, *, targets) -> tuple[np.ndarray, ...]``).
                Stored as-is; registering again for the same type replaces
                the previous rule.

        Raises:
            TypeError: If ``channel_type`` is not a `Channel` subclass or
                ``rule`` is not callable.
        """
        if not (isinstance(channel_type, type) and issubclass(channel_type, Channel)):
            raise TypeError(f"expected a Channel subclass, got {channel_type!r}")
        if not callable(rule):
            raise TypeError(
                f"rule for {channel_type.__name__} must be callable, got {rule!r}"
            )
        self._rules[channel_type] = rule

    def get(
        self, channel_type: type[Channel]
    ) -> ChannelImplementation | Callable | None:
        """Return the rule for a descriptor type, or ``None`` if unsupported."""
        return self._rules.get(channel_type)

    def supported_channels(self) -> frozenset[type[Channel]]:
        """Return every descriptor type with a registered rule."""
        return frozenset(self._rules)

    def copy(self) -> "ChannelImplementationMap":
        """Return a new map with an independent copy of the registrations.

        Rule objects themselves are shared (rules are expected to be pure);
        mutating one map's registrations never affects the other.
        """
        clone = ChannelImplementationMap()
        clone._rules = dict(self._rules)
        return clone


@dataclass(frozen=True)
class NoiseSupportReport:
    """Backend verdict on a `NoiseModel`, per noise source.

    Attributes:
        supported: ``True`` when every source in the model is executable.
        accepted_sources: Names of sources the backend can execute.
        rejected_sources: Names of sources the backend cannot execute.
        warnings: Human-readable notes accompanying the rejections.
    """

    supported: bool
    accepted_sources: tuple[str, ...] = ()
    rejected_sources: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _validate_cptp(kraus_ops: tuple[np.ndarray, ...], dim: int, label: str) -> None:
    """Validate a resolved Kraus tuple: shapes and CPTP completeness.

    Checked at the resolution site, right after the rule returns - a bare
    tuple of arrays cannot validate itself, and the resolution site has both
    the data and the context (target dimension, channel name) to do it.

    Args:
        kraus_ops: Resolved Kraus operators.
        dim: Combined dimension of the targeted subsystems.
        label: Channel name used in error messages.

    Raises:
        BackendValidationError: If the tuple is empty, an operator is not a
            ``(dim, dim)`` matrix, or completeness ``sum_i K_i^H K_i = I``
            fails within tolerance.
    """
    if len(kraus_ops) == 0:
        raise BackendValidationError(f"{label} resolved to an empty Kraus tuple")
    for kraus in kraus_ops:
        if not isinstance(kraus, np.ndarray) or kraus.shape != (dim, dim):
            raise BackendValidationError(
                f"{label} resolved to a Kraus operator of shape "
                f"{getattr(kraus, 'shape', type(kraus))}, expected {(dim, dim)}"
            )
    completeness = sum(kraus.conj().T @ kraus for kraus in kraus_ops)
    if not np.allclose(completeness, np.eye(dim)):
        raise BackendValidationError(
            f"{label} is not trace-preserving: sum K^H K != I for its "
            "resolved Kraus operators"
        )
