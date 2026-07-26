"""Built-in channel descriptors and their Kraus-producing rules.

Descriptors and rules are paired in this module the same way gate matrices
and their rules pair in ``implementation.matrices``. All entries resolve at
the register dimension of their actual targets, read inside the rule - one
``Depolarizing(p=0.01)`` instance is reusable on any dimension.

Arity is a property of the descriptor class: `Depolarizing` acts jointly on
however many subsystems the gate it is attached to targets; `AmplitudeDamping`
and `PhaseDamping` are single-subsystem channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, prod
from typing import ClassVar

import numpy as np

from ..errors import BackendValidationError
from ..implementation.matrices import clock_matrix, shift_matrix
from ..registers import RegisterRef
from .base import Channel


def _require_probability(value: float, label: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a real number in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number in [0, 1], got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class Depolarizing(Channel):
    """Uniform depolarizing channel ``rho -> (1-p) rho + p I/d``.

    Acts jointly on all subsystems of the gate occurrence it is attached to:
    ``d`` is the combined dimension of the targets, so the same descriptor
    depolarizes a single qubit, a qutrit, or the joint space of a two-qubit
    gate alike.

    Attributes:
        p: Probability of full depolarization, in ``[0, 1]``.
    """

    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "Depolarizing.p")


def depolarizing_rule(
    channel: Depolarizing, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `Depolarizing` into ``d**2`` Weyl-basis Kraus operators.

    Uses the identity ``(1/d**2) sum_{a,b} U_ab rho U_ab^H = I/d`` for the
    Weyl operators ``U_ab = Shift^a Clock^b`` (a unitary 1-design in any
    dimension, including the composite dimension of a multi-subsystem
    target), which makes the mixture below exactly ``(1-p) rho + p I/d``.
    At ``d=2`` this reduces to the textbook single-qubit channel: ``Shift=X``,
    ``Clock=Z``, and ``X@Z`` is ``Y`` up to a global phase that cancels in
    ``K rho K^H``.
    """
    p = channel.p
    dim = prod(ref.register.dim for ref in targets)
    ops = [np.sqrt(1 - p + p / dim**2) * np.eye(dim, dtype=complex)]
    for a in range(dim):
        for b in range(dim):
            if (a, b) == (0, 0):
                continue
            weyl = shift_matrix(dim, a) @ clock_matrix(dim, b)
            ops.append(np.sqrt(p / dim**2) * weyl)
    return tuple(ops)


@dataclass(frozen=True)
class AmplitudeDamping(Channel):
    """Amplitude damping: sequential (ladder) decay toward the ground state.

    Level ``k`` decays only to level ``k - 1`` with its own rate - the
    standard model for cascaded atomic relaxation and single-photon loss.
    Single-subsystem.

    Attributes:
        gammas: One decay rate per adjacent-level transition, each in
            ``[0, 1]``; ``gammas[k - 1]`` is the decay rate from level ``k``
            to ``k - 1``. A qubit takes ``gammas=(gamma,)``; dimension ``d``
            requires ``d - 1`` rates (checked at resolution, where the target
            dimension is known).
    """

    _num_subsystems: ClassVar[int | None] = 1
    gammas: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gammas", tuple(self.gammas))
        if len(self.gammas) == 0:
            raise ValueError("AmplitudeDamping requires at least one decay rate")
        for k, gamma in enumerate(self.gammas):
            _require_probability(gamma, f"AmplitudeDamping.gammas[{k}]")


def amplitude_damping_rule(
    channel: AmplitudeDamping, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `AmplitudeDamping` into its two ladder-decay Kraus operators.

    ``K0`` scales each level's survival amplitude; ``K1`` moves each level one
    step down. Completeness holds for any rates in ``[0, 1]``:
    ``K0^H K0 + K1^H K1 = I``.

    Raises:
        BackendValidationError: If the channel is applied to more than one
            subsystem, or the number of rates does not match the target
            dimension.
    """
    _require_channel_arity(channel, targets, "AmplitudeDamping")
    dim = targets[0].register.dim
    gammas = channel.gammas
    if len(gammas) != dim - 1:
        raise BackendValidationError(
            f"AmplitudeDamping needs {dim - 1} decay rate(s) for dimension "
            f"{dim}, got {len(gammas)}"
        )
    k0 = np.diag([1.0] + [np.sqrt(1 - g) for g in gammas]).astype(complex)
    k1 = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        k1[k - 1, k] = np.sqrt(gammas[k - 1])
    return (k0, k1)


@dataclass(frozen=True)
class PhaseDamping(Channel):
    """Phase damping (dephasing): coherence decay with no population transfer.

    Single-subsystem. Every diagonal entry of ``rho`` is preserved exactly;
    at ``d=2`` off-diagonal coherence survives at factor ``1 - p``.

    Attributes:
        p: Probability of full dephasing, in ``[0, 1]``.
    """

    _num_subsystems: ClassVar[int | None] = 1
    p: float

    def __post_init__(self) -> None:
        _require_probability(self.p, "PhaseDamping.p")


def phase_damping_rule(
    channel: PhaseDamping, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `PhaseDamping` into ``d`` Clock-power Kraus operators.

    Restricting the depolarizing construction to the diagonal ``Clock``
    powers gives the unique channel invariant under the diagonal subgroup:
    populations are untouched (every ``Clock^m`` is diagonal unitary) while
    coherences decay.

    Raises:
        BackendValidationError: If the channel is applied to more than one
            subsystem.
    """
    _require_channel_arity(channel, targets, "PhaseDamping")
    p = channel.p
    dim = targets[0].register.dim
    ops = [np.sqrt(1 - p + p / dim) * np.eye(dim, dtype=complex)]
    for m in range(1, dim):
        ops.append(np.sqrt(p / dim) * clock_matrix(dim, m))
    return tuple(ops)


def _require_channel_arity(
    channel: Channel, targets: tuple[RegisterRef, ...], label: str
) -> None:
    """Defensively enforce a fixed descriptor arity for direct rule callers."""
    expected = channel.num_subsystems
    if expected is None or len(targets) == expected:
        return
    if expected == 1:
        raise BackendValidationError(
            f"{label} is a single-subsystem channel; it cannot be attached to "
            f"an operation targeting {len(targets)} subsystems"
        )
    raise BackendValidationError(
        f"{label} is a {expected}-subsystem channel; it cannot be attached to "
        f"an extent of {len(targets)} subsystems"
    )


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

    Attach both returned channels to the same extent: a single-qubit gate
    occurrence, or one slot of a multi-qubit gate via ``add_noise(...,
    slots=)``. Order does not matter; the two commute.

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
