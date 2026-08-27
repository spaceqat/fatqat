"""Built-in channel descriptors and their Kraus-producing rules.

Descriptors and rules are paired in this module the same way gate matrices
and their rules pair in ``implementation.matrices``. All entries resolve at
the register dimension of their actual targets, read inside the rule - one
``Depolarizing(p=0.01)`` instance is reusable on any dimension.

Arity usually belongs to the descriptor class. Probability-form
`Depolarizing` is width-agnostic while rate form is local to one subsystem;
`AmplitudeDamping` and `PhaseDamping` are single-subsystem channels.
`PauliChannel` derives its per-instance arity from its Pauli-string width.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass
from math import expm1, isfinite, log1p, prod
from typing import ClassVar

import numpy as np

from ..errors import BackendValidationError
from ..implementation.matrices import _I, _X, _Y, _Z, clock_matrix, shift_matrix
from ..registers import RegisterRef
from .base import Channel


def _require_probability(value: float, label: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a real number in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number in [0, 1], got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value!r}")


def _require_rate(value: float, label: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite, non-negative real."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} must be a finite, non-negative real number, got {value!r}"
        )
    if not isfinite(value) or value < 0.0:
        raise ValueError(
            f"{label} must be a finite, non-negative real number, got {value!r}"
        )


def _require_duration(value: float, label: str = "duration") -> None:
    """Raise ``ValueError`` unless ``value`` is a finite, non-negative real."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} must be a finite, non-negative real number, got {value!r}"
        )
    if not isfinite(value) or value < 0.0:
        raise ValueError(
            f"{label} must be a finite, non-negative real number, got {value!r}"
        )


def _normalize_damping_values(value: object, label: str) -> tuple[float, ...]:
    """Normalize a scalar or iterable of numbers into a non-empty tuple.

    A bare number becomes a one-element tuple (one transition). A string is
    rejected even though it is iterable - it is never a legal element
    sequence here.
    """
    if isinstance(value, (int, float)):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(
            f"{label} must be a number or an iterable of numbers, got {value!r}"
        )
    normalized = tuple(value)
    if len(normalized) == 0:
        raise ValueError(f"{label} requires at least one value")
    return normalized


def _p_to_rate(p: float, duration: float) -> float:
    """Convert one finite-channel probability to a rate over ``duration``.

    Uses the numerically stable ``log1p`` form of ``rate = -log(1 - p) / t``,
    accurate when ``p`` is small (the common case for one gate occurrence).
    """
    _require_duration(duration)
    if p >= 1.0:
        raise ValueError(
            "probability 1 is a valid finite channel but has no finite rate"
        )
    if duration == 0.0:
        if p != 0.0:
            raise ValueError(
                "a nonzero probability has no finite rate at zero duration"
            )
        return 0.0
    return -log1p(-p) / duration


def _rate_to_p(rate: float, duration: float) -> float:
    """Convert one rate to a finite-channel probability over ``duration``.

    Uses the numerically stable ``expm1`` form of ``p = 1 - exp(-rate * t)``,
    accurate when ``rate * t`` is small.
    """
    _require_duration(duration)
    if duration == 0.0:
        return 0.0
    return -expm1(-rate * duration)


@dataclass(frozen=True, kw_only=True)
class Depolarizing(Channel):
    """Uniform depolarization in finite-probability or continuous-rate mode.

    Exactly one of ``p`` and ``rate`` must be supplied. Matrix backends consume
    probability mode over an operation occurrence. Pulse emulators consume
    rate mode as a local continuous generator; backends do not convert modes
    implicitly.

    Attributes:
        p: Probability of full depolarization for one finite channel
            application, in ``[0, 1]``.
        rate: Depolarization rate in the pulse backend's inverse time unit.

    Examples:
        >>> import fatqat as fq
        >>> fq.noise.Depolarizing(p=0.01).p
        0.01
        >>> fq.noise.Depolarizing(rate=0.002).rate
        0.002
    """

    p: float | None = None
    rate: float | None = None

    def __post_init__(self) -> None:
        if (self.p is None) == (self.rate is None):
            raise ValueError("Depolarizing requires exactly one of p or rate")
        if self.p is not None:
            _require_probability(self.p, "Depolarizing.p")
        else:
            _require_rate(self.rate, "Depolarizing.rate")

    @property
    def num_subsystems(self) -> int | None:
        """Return the declaration arity implied by the authored mode.

        Returns:
            ``1`` for local continuous-rate mode, or ``None`` for
            width-agnostic finite-probability mode.
        """
        return 1 if self.rate is not None else None

    def as_probability(self, duration: float) -> float:
        """Return this declaration's probability over a duration.

        Args:
            duration: Finite nonnegative elapsed time in the backend's unit.

        Returns:
            Authored ``p`` unchanged, or ``1 - exp(-rate * duration)``.

        Raises:
            ValueError: If ``duration`` is invalid.
        """
        if self.p is not None:
            _require_duration(duration)
            return self.p
        return _rate_to_p(self.rate, duration)

    def as_rate(self, duration: float) -> float:
        """Return this declaration's continuous rate over a duration.

        Args:
            duration: Finite nonnegative elapsed time in the backend's unit.

        Returns:
            Authored ``rate`` unchanged, or ``-log(1 - p) / duration``.

        Raises:
            ValueError: If ``duration`` is invalid or the finite probability
                has no finite rate for that duration.
        """
        if self.rate is not None:
            _require_duration(duration)
            return self.rate
        return _p_to_rate(self.p, duration)


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
    if p is None:
        raise BackendValidationError(
            "Depolarizing in rate mode has no matrix-backend Kraus "
            "implementation; use probability mode or a pulse backend"
        )
    dim = prod(ref.register.dim for ref in targets)
    ops = [np.sqrt(1 - p + p / dim**2) * np.eye(dim, dtype=complex)]
    for a in range(dim):
        for b in range(dim):
            if (a, b) == (0, 0):
                continue
            weyl = shift_matrix(dim, a) @ clock_matrix(dim, b)
            ops.append(np.sqrt(p / dim**2) * weyl)
    return tuple(ops)


_PAULI_MATRICES = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

_PAULI_PROBABILITY_TOL = 1e-9


def _pauli_string_matrix(string: str) -> np.ndarray:
    """Build a Pauli string's ``2**k x 2**k`` matrix, ``string[0]`` most-significant.

    ``string[0]`` describes ``targets[0]``, the matrix's most-significant index
    digit - the same convention as a gate matrix, and the reverse of Qiskit's
    ``Pauli("IX")`` reading.
    """
    matrix = _PAULI_MATRICES[string[0]]
    for letter in string[1:]:
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def _normalize_pauli_terms(
    terms: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    """Normalize a Pauli-term mapping or pair sequence into canonical form.

    Canonical form leads with the all-identity term carrying the probability
    the other terms leave unassigned, followed by the non-identity terms in the
    order given. An explicit identity entry must agree with that implied value.
    """
    items = tuple(terms.items()) if isinstance(terms, Mapping) else tuple(terms)
    if not items:
        raise ValueError("PauliChannel requires at least one term")

    width = -1
    seen: set[str] = set()
    for entry in items:
        if len(entry) != 2:
            raise ValueError(
                f"PauliChannel term must be a (string, p) pair, got {entry!r}"
            )
        string, p = entry
        if not isinstance(string, str) or not string:
            raise ValueError(
                f"PauliChannel term label must be a non-empty string, got {string!r}"
            )
        if any(letter not in _PAULI_MATRICES for letter in string):
            raise ValueError(
                f"PauliChannel term {string!r} must use only the letters I, X, Y, Z"
            )
        if width < 0:
            width = len(string)
        elif len(string) != width:
            raise ValueError(
                f"PauliChannel terms must all be the same width; got {len(string)} "
                f"for {string!r} after {width}"
            )
        if string in seen:
            raise ValueError(f"PauliChannel term {string!r} is registered twice")
        seen.add(string)
        _require_probability(p, f"PauliChannel[{string}]")

    identity = "I" * width
    others = tuple((string, float(p)) for string, p in items if string != identity)
    assigned = sum(p for _, p in others)
    if assigned > 1.0 + _PAULI_PROBABILITY_TOL:
        raise ValueError(
            f"PauliChannel error probabilities sum to {assigned}, which exceeds 1"
        )
    implied = max(0.0, 1.0 - assigned)
    for string, p in items:
        if string == identity and abs(p - implied) > _PAULI_PROBABILITY_TOL:
            raise ValueError(
                f"PauliChannel identity probability {p} conflicts with the "
                f"{implied} its other terms leave unassigned"
            )
    return ((identity, implied),) + others


@dataclass(frozen=True)
class PauliChannel(Channel):
    """A stochastic Pauli channel: ``rho -> sum_i p_i P_i rho P_i``.

    Each term names a Pauli string over the subsystems the channel is attached
    to and the probability that error occurs.

    Qubits only (``dim == 2``), checked at resolution time; use `Depolarizing`
    for the dimension-generic uniform channel.

    Attributes:
        terms: ``(pauli_string, probability)`` pairs, the all-identity term
            first. Accepts a mapping or a sequence of pairs; the width of the
            strings sets the channel's arity, ``string[0]`` describing the
            first target.

    Probabilities need not sum to 1: whatever the error terms leave unassigned
    becomes the probability of the all-identity (no-error) term. Stating that
    term explicitly is allowed, but only with the value the others imply.

    Examples:
        A biased single-qubit channel, 1% X and 2% Z:

        >>> import fatqat as fq
        >>> fq.noise.PauliChannel({"X": 0.01, "Z": 0.02}).terms
        (('I', 0.97), ('X', 0.01), ('Z', 0.02))

        A correlated two-qubit channel for a CX:

        >>> fq.noise.PauliChannel({"XX": 0.005, "ZI": 0.01}).num_subsystems
        2
    """

    terms: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", _normalize_pauli_terms(self.terms))

    @property
    def num_subsystems(self) -> int:
        """Number of qubits the channel acts on: the width of its Pauli strings."""
        return len(self.terms[0][0])


def pauli_channel_rule(
    channel: PauliChannel, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `PauliChannel` into one ``sqrt(p_i) P_i`` Kraus operator per term.

    Completeness is immediate: every ``P_i`` is unitary, so
    ``sum_i K_i^H K_i = (sum_i p_i) I = I``.

    Raises:
        BackendValidationError: If the target count does not match the term
            width, or any target is not a qubit.
    """
    _require_channel_arity(channel, targets, "PauliChannel")
    for ref in targets:
        if ref.register.dim != 2:
            raise BackendValidationError(
                "PauliChannel is defined on qubits only, but a target has "
                f"dimension {ref.register.dim}; use Depolarizing or PhaseDamping "
                "for a dimension-generic channel"
            )
    return tuple(
        np.sqrt(p) * _pauli_string_matrix(string) for string, p in channel.terms
    )


@dataclass(frozen=True, kw_only=True)
class AmplitudeDamping(Channel):
    """Amplitude damping: sequential (ladder) decay toward the ground state.

    Level ``k`` decays only to level ``k - 1`` with its own probability or
    rate - the standard model for cascaded atomic relaxation and
    single-photon loss. Single-subsystem.

    Exactly one of ``p`` and ``rate`` must be given, keyword-only, each as a
    scalar (one adjacent-level transition) or a tuple (one value per
    transition; ``value[k - 1]`` describes the level ``k`` to ``k - 1``
    transition). Dimension ``d`` requires ``d - 1`` values, checked at
    resolution time, where the target dimension is known.

    Attributes:
        p: Decay probability per transition for one finite channel
            application, each in ``[0, 1]``.
        rate: Decay rate per transition, in the inverse of the target
            backend's declared time unit. Convert with
            :py:meth:`as_probability`.
    """

    num_subsystems: ClassVar[int | None] = 1
    p: tuple[float, ...] | None = None
    rate: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (self.p is None) == (self.rate is None):
            raise ValueError("AmplitudeDamping requires exactly one of p or rate")
        if self.p is not None:
            values = _normalize_damping_values(self.p, "AmplitudeDamping.p")
            for k, value in enumerate(values):
                _require_probability(value, f"AmplitudeDamping.p[{k}]")
            object.__setattr__(self, "p", values)
        else:
            values = _normalize_damping_values(self.rate, "AmplitudeDamping.rate")
            for k, value in enumerate(values):
                _require_rate(value, f"AmplitudeDamping.rate[{k}]")
            object.__setattr__(self, "rate", values)

    def as_probability(self, duration: float) -> tuple[float, ...]:
        """Return this channel's per-transition probabilities over ``duration``.

        Returns ``p`` unchanged if already in probability mode (``duration``
        is still validated); otherwise converts each rate.
        """
        if self.p is not None:
            _require_duration(duration)
            return self.p
        return tuple(_rate_to_p(rate, duration) for rate in self.rate)

    def as_rate(self, duration: float) -> tuple[float, ...]:
        """Return this channel's per-transition rates over ``duration``.

        Returns ``rate`` unchanged if already in rate mode (``duration`` is
        still validated); otherwise converts each probability.
        """
        if self.rate is not None:
            _require_duration(duration)
            return self.rate
        return tuple(_p_to_rate(p, duration) for p in self.p)


def amplitude_damping_rule(
    channel: AmplitudeDamping, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `AmplitudeDamping` into its two ladder-decay Kraus operators.

    ``K0`` scales each level's survival amplitude; ``K1`` moves each level one
    step down. Completeness holds for any probabilities in ``[0, 1]``:
    ``K0^H K0 + K1^H K1 = I``.

    Raises:
        BackendValidationError: If the channel is in rate mode (no matrix
            Kraus implementation exists for a rate without a duration), is
            applied to more than one subsystem, or its value count does not
            match the target dimension.
    """
    _require_channel_arity(channel, targets, "AmplitudeDamping")
    if channel.p is None:
        raise BackendValidationError(
            "AmplitudeDamping in rate mode has no matrix-backend Kraus "
            "implementation; use probability mode or a duration-aware backend"
        )
    dim = targets[0].register.dim
    ps = channel.p
    if len(ps) != dim - 1:
        raise BackendValidationError(
            f"AmplitudeDamping needs {dim - 1} p value(s) for dimension "
            f"{dim}, got {len(ps)}"
        )
    k0 = np.diag([1.0] + [np.sqrt(1 - p) for p in ps]).astype(complex)
    k1 = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        k1[k - 1, k] = np.sqrt(ps[k - 1])
    return (k0, k1)


@dataclass(frozen=True, kw_only=True)
class PhaseDamping(Channel):
    """Phase damping (dephasing): coherence decay with no population transfer.

    Single-subsystem, scalar-valued. Every diagonal entry of ``rho`` is
    preserved exactly; at ``d=2`` off-diagonal coherence survives at factor
    ``1 - p``.

    Exactly one of ``p``, ``rate``, and ``t_phi`` must be given,
    keyword-only. ``t_phi`` is normalized immediately to
    ``rate = 1 / t_phi``.

    Attributes:
        p: Probability of full dephasing for one finite channel application,
            in ``[0, 1]``.
        rate: Dephasing rate, in the inverse of the target backend's declared
            time unit. Convert with :py:meth:`as_probability`.
    """

    num_subsystems: ClassVar[int | None] = 1
    p: float | None = None
    rate: float | None = None
    t_phi: InitVar[float | None] = None

    def __post_init__(self, t_phi: float | None) -> None:
        if sum(value is not None for value in (self.p, self.rate, t_phi)) != 1:
            raise ValueError("PhaseDamping requires exactly one of p, rate, or t_phi")
        if self.p is not None:
            _require_probability(self.p, "PhaseDamping.p")
        elif self.rate is not None:
            _require_rate(self.rate, "PhaseDamping.rate")
        else:
            if isinstance(t_phi, bool) or not isinstance(t_phi, (int, float)):
                raise ValueError(
                    "PhaseDamping.t_phi must be a finite, positive real number, "
                    f"got {t_phi!r}"
                )
            if not isfinite(t_phi) or t_phi <= 0.0:
                raise ValueError(
                    "PhaseDamping.t_phi must be a finite, positive real number, "
                    f"got {t_phi!r}"
                )
            object.__setattr__(self, "rate", 1.0 / t_phi)

    def as_probability(self, duration: float) -> float:
        """Return this channel's probability over ``duration``.

        Returns ``p`` unchanged if already in probability mode (``duration``
        is still validated); otherwise converts the rate.
        """
        if self.p is not None:
            _require_duration(duration)
            return self.p
        return _rate_to_p(self.rate, duration)

    def as_rate(self, duration: float) -> float:
        """Return this channel's rate over ``duration``.

        Returns ``rate`` unchanged if already in rate mode (``duration`` is
        still validated); otherwise converts the probability.
        """
        if self.rate is not None:
            _require_duration(duration)
            return self.rate
        return _p_to_rate(self.p, duration)


def phase_damping_rule(
    channel: PhaseDamping, *, targets: tuple[RegisterRef, ...]
) -> tuple[np.ndarray, ...]:
    """Resolve `PhaseDamping` into ``d`` Clock-power Kraus operators.

    Restricting the depolarizing construction to the diagonal ``Clock``
    powers gives the unique channel invariant under the diagonal subgroup:
    populations are untouched (every ``Clock^m`` is diagonal unitary) while
    coherences decay.

    Raises:
        BackendValidationError: If the channel is in rate mode (no matrix
            Kraus implementation exists for a rate without a duration), or is
            applied to more than one subsystem.
    """
    _require_channel_arity(channel, targets, "PhaseDamping")
    if channel.p is None:
        raise BackendValidationError(
            "PhaseDamping in rate mode has no matrix-backend Kraus "
            "implementation; use probability mode or a duration-aware backend"
        )
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
