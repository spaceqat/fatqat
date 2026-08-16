"""Channel-noise abstraction: the descriptor marker, the rule protocol, the
rule registry, and the resolved-Kraus shape check.

A channel descriptor (`Channel` subclass) carries physical parameters only,
never arrays - the same way an :py:class:`~fatqat.operations.Operation` like
``RX(0.3)`` never stores its matrix. A `ChannelImplementation` rule turns a
descriptor plus its resolution-time ``targets`` into a bare tuple of Kraus
arrays, mirroring how a matrix-implementation rule produces a bare matrix.
`ChannelImplementationMap` is the matrix family's registry from descriptor
type to rule; it exists (instead of reusing the gate implementation map)
because a channel resolves to a *tuple* of Kraus operators here but to local
Lindblad operators in continuous simulators (see ``noise.lindblad``) - the
same descriptor means different mathematical objects per backend family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np

from ..errors import BackendValidationError
from ..registers import RegisterRef


class Channel:
    """Base marker for channel descriptors.

    Concrete subclasses (see ``noise.catalog``) are frozen dataclasses holding
    physical parameters only - rates, probabilities - never Kraus arrays. The
    array computation belongs entirely to the `ChannelImplementation` rule
    registered for the subclass. A subclass may declare ``_num_subsystems``
    when it has a fixed positive arity; ``None`` means it is width-agnostic.
    """

    _num_subsystems: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        arity = cls._num_subsystems
        if arity is not None and (
            not isinstance(arity, int) or isinstance(arity, bool) or arity < 1
        ):
            raise ValueError(
                f"_num_subsystems must be a positive int or None, got {arity!r}"
            )

    @property
    def num_subsystems(self) -> int | None:
        """Number of subsystems this channel acts on, or ``None`` for any width."""
        return type(self)._num_subsystems


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


class _ChannelImplementationRegistry:
    """Shared class-keyed registration mechanics for channel implementations."""

    def __init__(self) -> None:
        self._rules: dict[type[Channel], ChannelImplementation | Callable] = {}

    def register(
        self,
        channel_type: type[Channel],
        rule: ChannelImplementation | Callable,
    ) -> None:
        """Register one channel descriptor implementation rule.

        Args:
            channel_type: `Channel` subclass to key the registry by.
            rule: Backend-specific callable stored as-is; registering again
                for the same type replaces the previous rule.

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

    def copy(self) -> "_ChannelImplementationRegistry":
        """Return a new map with an independent copy of the registrations.

        Rule objects themselves are shared (rules are expected to be pure);
        mutating one map's registrations never affects the other.
        """
        clone = type(self)()
        clone._rules = dict(self._rules)
        return clone


class ChannelImplementationMap(_ChannelImplementationRegistry):
    """Resolve channel descriptor types to their Kraus-producing rules.

    A dumb class-keyed registry: it never invokes rules itself and holds no
    layout or payload knowledge. The map is static per backend instance -
    which `NoiseModel` a run uses has no bearing on what the map can resolve.
    Its coverage is the backend's channel capability declaration: a descriptor
    type without a registered rule is unsupported.
    """


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


# Round-off slack when testing ``K^H K`` for a multiple of the identity.
_UNITARY_BRANCH_ATOL = 1e-12


def _unitary_branch_probabilities(
    kraus_ops: tuple[np.ndarray, ...],
) -> np.ndarray | None:
    """Branch probabilities of a channel of scaled unitaries, else ``None``.

    Every operator must satisfy ``K_i^H K_i = p_i I``, which makes the branch
    probability ``<psi|K_i^H K_i|psi> = p_i`` independent of the state. The
    test is on operator content, not on descriptor type: `Depolarizing` and
    `PhaseDamping` satisfy it in any dimension, `AmplitudeDamping` does not.

    Returns the ``p_i`` in Kraus order, unnormalized; ``None`` when any
    operator fails the test or the channel is degenerately all-zero.
    """
    probabilities = np.empty(len(kraus_ops), dtype=float)
    for i, kraus in enumerate(kraus_ops):
        gram = kraus.conj().T @ kraus
        dim = gram.shape[0]
        p = float(np.real(np.trace(gram))) / dim
        if not np.allclose(gram, p * np.eye(dim), rtol=0.0, atol=_UNITARY_BRANCH_ATOL):
            return None
        probabilities[i] = max(p, 0.0)
    if probabilities.sum() <= 0.0:
        return None
    return probabilities


def _sampled_unitary_branches(
    kraus_ops: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[bool, ...]] | None:
    """Everything a sampler needs to draw one branch without touching the state.

    Returns ``(probabilities, unitaries, is_identity)`` for a scaled-unitary
    channel, or ``None`` when `_unitary_branch_probabilities` rejects it.
    ``unitaries`` are the operators divided by their own scale, so applying a
    drawn one preserves the norm; ``is_identity`` marks the branches a sampler
    can skip outright. A zero-probability branch keeps its operator unscaled
    and can never be drawn.
    """
    probabilities = _unitary_branch_probabilities(kraus_ops)
    if probabilities is None:
        return None
    unitaries: list[np.ndarray] = []
    identities: list[bool] = []
    for kraus, p in zip(kraus_ops, probabilities):
        unitary = kraus / np.sqrt(p) if p > 0.0 else kraus
        unitaries.append(unitary)
        identities.append(
            bool(
                np.allclose(
                    unitary,
                    np.eye(unitary.shape[0]),
                    rtol=0.0,
                    atol=_UNITARY_BRANCH_ATOL,
                )
            )
        )
    return probabilities, tuple(unitaries), tuple(identities)


def _validate_kraus_shapes(
    kraus_ops: tuple[np.ndarray, ...], dim: int, label: str
) -> None:
    """Validate a resolved Kraus tuple's shapes against the target dimension.

    Checked at the resolution site, right after the rule returns - a bare
    tuple of arrays cannot validate itself, and the resolution site has both
    the data and the context (target dimension, channel name) to do it.

    Deliberately shape-only: CPTP completeness of the built-in catalog is
    covered by its tests, and user-supplied rules are not required to be
    trace-preserving at runtime - the same posture as gate matrices, which
    are never checked for unitarity. An unphysical channel may be
    meaningless, but it is not an error.

    Args:
        kraus_ops: Resolved Kraus operators.
        dim: Combined dimension of the targeted subsystems.
        label: Channel name used in error messages.

    Raises:
        BackendValidationError: If the tuple is empty or an operator is not a
            ``(dim, dim)`` matrix.
    """
    if len(kraus_ops) == 0:
        raise BackendValidationError(f"{label} resolved to an empty Kraus tuple")
    for kraus in kraus_ops:
        if not isinstance(kraus, np.ndarray) or kraus.shape != (dim, dim):
            raise BackendValidationError(
                f"{label} resolved to a Kraus operator of shape "
                f"{getattr(kraus, 'shape', type(kraus))}, expected {(dim, dim)}"
            )
