"""Public descriptors and implementation maps for quantum-channel noise."""

from __future__ import annotations

from typing import Callable, ClassVar, Self

import numpy as np

from ..errors import BackendValidationError
from ..registers import RegisterRef


class Channel:
    """Base class for backend-independent quantum noise.

    A channel stores parameters such as probabilities, rates, and times. A
    simulator resolves its exact type through a `ChannelImplementationMap`;
    pulse emulators use family-owned Lindblad realizations. Registering a rule
    for a base class does not implement its subclasses.

    Set ``num_subsystems`` to a positive ``int`` other than ``bool`` when every
    instance acts on a fixed number of subsystems. Leave it as ``None`` to use
    the matched operation's width, or expose a property when instance data
    determines the width. Built-in channels are immutable; treat custom
    channels the same way after adding them to a noise model.

    Attributes:
        num_subsystems: Fixed number of targeted subsystems, or ``None`` when
            the channel takes its width from the matched operation.
    """

    num_subsystems: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "_num_subsystems" in cls.__dict__:
            raise TypeError(
                "_num_subsystems is no longer supported on Channel "
                "subclasses; declare num_subsystems instead"
            )
        arity = cls.num_subsystems
        # Some descriptors derive their arity from instance data and expose
        # it through a property under the same public name.
        if isinstance(arity, property):
            return
        if arity is not None and (
            not isinstance(arity, int) or isinstance(arity, bool) or arity < 1
        ):
            raise ValueError(
                f"num_subsystems must be a positive int or None, got {arity!r}"
            )


class ChannelImplementation:
    """Optional base class for a simulator channel rule.

    A matrix rule is called as ``rule(channel, *, targets=targets)``. The
    ordered ``targets`` are the matched program ``RegisterRef`` objects, so a
    rule can use their register dimensions to build its operators. The rule
    returns a nonempty tuple of NumPy Kraus matrices. Each matrix must have
    shape ``(D, D)``, where ``D`` is the product of the target dimensions.

    The backend checks the number and shapes of the returned arrays, but not
    complete positivity or trace preservation. The rule is responsible for
    those physical guarantees. A plain callable with the same signature is
    usually enough; subclass this type only for a configured rule object.
    """

    def __call__(
        self, channel: Channel, *, targets: tuple[RegisterRef, ...]
    ) -> tuple[np.ndarray, ...]:
        """Return Kraus operators for one channel application.

        Args:
            channel: Noise object being resolved.
            targets: Ordered program targets for the matching operation.

        Returns:
            A nonempty tuple of ``(D, D)`` NumPy arrays.

        Raises:
            NotImplementedError: Always on this base implementation. A
                subclass must implement the rule.
        """
        raise NotImplementedError


class _ChannelImplementationRegistry:
    """Shared class-keyed registration mechanics for channel implementations."""

    def __init__(self) -> None:
        self._rules: dict[type[Channel], ChannelImplementation | Callable] = {}

    def add(
        self,
        channel_type: type[Channel],
        rule: ChannelImplementation | Callable,
    ) -> None:
        """Register or replace the rule for one exact channel type.

        Lookup uses the concrete channel type and never searches its base
        classes. The callable is stored by reference and its signature is
        checked only when a backend invokes it.

        Args:
            channel_type: `Channel` subclass used as the exact lookup key.
            rule: Backend-specific callable. Adding the same type again
                replaces its previous rule.

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
        """Return the rule registered for exactly ``channel_type``.

        The lookup does not fall back to a registered base class.

        Args:
            channel_type: Exact channel type to look up.

        Returns:
            The stored callable, or ``None`` when the type is not registered.
        """
        return self._rules.get(channel_type)

    def supported_channels(self) -> frozenset[type[Channel]]:
        """Return an immutable snapshot of the registered channel types."""
        return frozenset(self._rules)

    def copy(self) -> Self:
        """Return a map whose registrations can be changed independently."""
        clone = type(self)()
        clone._rules = dict(self._rules)
        return clone


class ChannelImplementationMap(_ChannelImplementationRegistry):
    """Map exact channel types to simulator rules.

    A registered rule has the call shape ``rule(channel, *, targets=targets)``
    and returns a nonempty tuple of Kraus matrices. For targets with combined
    dimension ``D``, every result must be a NumPy array of shape ``(D, D)``.
    FATQAT validates these structural requirements but does not check that a
    custom rule is completely positive or trace preserving.

    A registered rule does not by itself guarantee backend support. A backend
    may still reject the channel's parameter form, scope, target dimensions,
    or execution method.
    """


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
