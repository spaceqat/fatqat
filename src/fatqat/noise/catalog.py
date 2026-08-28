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

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from math import expm1, isfinite, log1p, prod
from typing import ClassVar, cast

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
    """Convert one simulator-channel probability to a rate over ``duration``.

    Uses the numerically stable ``log1p`` form of ``rate = -log(1 - p) / t``,
    accurate when ``p`` is small (the common case for one gate application).
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
    """Convert one rate to a simulator-channel probability over ``duration``.

    Uses the numerically stable ``expm1`` form of ``p = 1 - exp(-rate * t)``,
    accurate when ``rate * t`` is small.
    """
    _require_duration(duration)
    if duration == 0.0:
        return 0.0
    return -expm1(-rate * duration)


@dataclass(frozen=True, kw_only=True)
class Depolarizing(Channel):
    """Uniform depolarizing noise for simulators or emulators.

    Pass exactly one of p and rate, by keyword. With p, a simulator applies
    ``rho -> (1 - p) rho + p I / d`` jointly to the selected operands, whose
    combined dimension is d. With rate, an emulator uses local Lindblad
    operators with evolution
    ``rate * (trace(rho) I / d - rho)`` on one subsystem.

    Backends do not convert between the two forms. Matrix simulators require p
    on an operation; a pulse emulator accepts rate only when its Lindblad map
    supports this noise type.

    Args:
        p: Probability of complete depolarization in one simulator
            application. Must be a finite ``int`` or ``float`` other than
            ``bool`` in ``[0, 1]``.
        rate: Depolarization rate for emulator Lindblad operators. Must be a
            finite nonnegative ``int`` or ``float`` other than ``bool``, in the
            inverse of the backend's time unit.

    Raises:
        ValueError: If both arguments or neither argument is supplied, or if a
            supplied value is outside its accepted range.

    Attributes:
        p: Provided simulator probability, or ``None`` in rate mode.
        rate: Provided emulator rate, or ``None`` in probability mode.

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
        """Return how many subsystems this form acts on.

        Returns:
            ``1`` for local emulator rate mode, or ``None`` for
            width-agnostic simulator probability mode.
        """
        return 1 if self.rate is not None else None

    def as_probability(self, duration: float) -> float:
        """Return the equivalent probability for an explicit duration.

        In probability mode, this returns the provided p unchanged after
        validating duration. In rate mode, it returns
        ``1 - exp(-rate * duration)``. No backend calls this conversion
        implicitly.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the same time unit whose inverse is used by rate.

        Returns:
            The provided probability or the converted rate-mode probability.

        Raises:
            ValueError: If duration is not a finite nonnegative real number.
        """
        if self.p is not None:
            _require_duration(duration)
            return self.p
        return _rate_to_p(self.rate, duration)

    def as_rate(self, duration: float) -> float:
        """Return the equivalent emulator rate for an explicit duration.

        In rate mode, this returns the provided rate unchanged after validating
        duration. In probability mode, it returns
        ``-log(1 - p) / duration``. No backend calls this conversion
        implicitly.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the time unit for the returned rate.

        Returns:
            The provided rate or the converted probability-mode rate.

        Raises:
            ValueError: If duration is invalid, if p is 1, or if a nonzero p
                is converted at zero duration.
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
    terms: Mapping[str, float] | Iterable[tuple[str, float]],
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
    """A stochastic mixture of Pauli errors.

    The channel acts as ``rho -> sum_i p_i P_i rho P_i``. Construct it from a
    mapping from Pauli strings to probabilities or from an iterable of
    ``(string, probability)`` pairs. Every string must be nonempty, use only
    the uppercase letters ``I``, ``X``, ``Y``, and ``Z``, and have the same
    width. The width sets the number of qubit targets.

    Strings follow target order: the first character describes the first
    target and is the most-significant factor in the local matrix. This is the
    reverse of Qiskit's displayed Pauli-string convention.

    Each probability must be a finite ``int`` or ``float`` other than ``bool``
    in ``[0, 1]``. Error terms may sum to less than 1; the unassigned weight
    becomes the all-identity probability. An explicit all-identity term is
    accepted only when it agrees with that implied value. Duplicate strings
    and a total error probability greater than 1 are rejected. Sum and
    identity comparisons allow ordinary floating-point round-off.

    FATQAT consumes the input and stores an immutable tuple with the
    all-identity term first, followed by nonidentity terms in input order.
    Changing an input mapping or sequence later has no effect. The channel
    requires one qubit target per character and works with simulators only.

    Args:
        terms: Pauli probabilities as a mapping or iterable of pairs.

    Raises:
        TypeError: If terms is not a mapping or iterable, or an iterable entry
            does not support the required two-element pair shape.
        ValueError: If terms is empty, malformed, duplicated, uses invalid or
            unequal-width strings, contains an invalid probability, assigns
            too much error probability, or contradicts the implied identity
            probability.

    Attributes:
        terms: Normalized ``(string, probability)`` pairs with identity first.

    Examples:
        A biased single-qubit channel, 1% X and 2% Z:

        >>> import fatqat as fq
        >>> fq.noise.PauliChannel({"X": 0.01, "Z": 0.02}).terms
        (('I', 0.97), ('X', 0.01), ('Z', 0.02))

        A correlated two-qubit channel for a CX:

        >>> fq.noise.PauliChannel({"XX": 0.005, "ZI": 0.01}).num_subsystems
        2
    """

    terms: Mapping[str, float] | Iterable[tuple[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", _normalize_pauli_terms(self.terms))

    @property
    def num_subsystems(self) -> int:
        """Return the number of qubits named by each Pauli string."""
        terms = cast(tuple[tuple[str, float], ...], self.terms)
        return len(terms[0][0])


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
    terms = cast(tuple[tuple[str, float], ...], channel.terms)
    return tuple(np.sqrt(p) * _pauli_string_matrix(string) for string, p in terms)


@dataclass(frozen=True, kw_only=True)
class AmplitudeDamping(Channel):
    """Relaxation between adjacent energy levels toward the ground state.

    Pass exactly one of p and rate, by keyword. Each may be one real number or
    a nonempty iterable. For a d-level target, pass ``d - 1`` values ordered as
    1 -> 0, 2 -> 1, and so on. A scalar therefore works only for a two-level
    target. FATQAT stores the values as a tuple and checks the length when it
    knows the target dimension.

    A simulator uses p for one channel application; each value must be an
    ``int`` or ``float`` other than ``bool`` in ``[0, 1]``. An emulator uses
    rate in a local ladder-transition Lindblad operator; each value must have
    the same numeric type, be finite and nonnegative, and use the inverse of
    the backend's time unit. Backends do not convert between forms. The noise
    acts on one subsystem.

    The conversion methods convert each transition independently. For more
    than two levels, that elementwise conversion is not the exact result of
    multilevel Lindblad-operator evolution, which can undergo more than one
    adjacent decay during an interval.

    Args:
        p: One simulator decay probability or an iterable of probabilities.
        rate: One emulator Lindblad-operator rate or an iterable of rates.

    Raises:
        ValueError: If both arguments or neither argument is supplied, an
            iterable is empty, or a value is not in its accepted range.

    Attributes:
        p: Normalized probability tuple, or ``None`` in rate mode.
        rate: Normalized rate tuple, or ``None`` in probability mode.

    Examples:
        >>> import fatqat as fq
        >>> fq.noise.AmplitudeDamping(p=0.01).p
        (0.01,)
        >>> fq.noise.AmplitudeDamping(rate=(0.001, 0.002)).rate
        (0.001, 0.002)
    """

    num_subsystems: ClassVar[int | None] = 1
    p: float | Iterable[float] | None = None
    rate: float | Iterable[float] | None = None

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
        """Return per-transition probabilities for an explicit duration.

        Probability mode returns the normalized p tuple unchanged after
        validating duration. Rate mode converts each entry with
        ``1 - exp(-rate * duration)``. The conversion is exact for a two-level
        channel; for a multilevel channel it is an elementwise utility, not the
        exact result of evolution under the full ladder-transition Lindblad
        operator.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the same time unit whose inverse is used by rate.

        Returns:
            One probability per adjacent-level transition.

        Raises:
            ValueError: If duration is not a finite nonnegative real number.
        """
        if self.p is not None:
            _require_duration(duration)
            return cast(tuple[float, ...], self.p)
        rates = cast(tuple[float, ...], self.rate)
        return tuple(_rate_to_p(rate, duration) for rate in rates)

    def as_rate(self, duration: float) -> tuple[float, ...]:
        """Return per-transition rates for an explicit duration.

        Rate mode returns the normalized rate tuple unchanged after validating
        duration. Probability mode converts each entry with
        ``-log(1 - p) / duration``. The conversion is exact for a two-level
        channel; for a multilevel channel it is an elementwise utility.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the time unit for the returned rates.

        Returns:
            One emulator rate per adjacent-level transition.

        Raises:
            ValueError: If duration is invalid, any p is 1, or a nonzero p is
                converted at zero duration.
        """
        if self.rate is not None:
            _require_duration(duration)
            return cast(tuple[float, ...], self.rate)
        probabilities = cast(tuple[float, ...], self.p)
        return tuple(_p_to_rate(p, duration) for p in probabilities)


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
    ps = cast(tuple[float, ...], channel.p)
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
    """Dephasing without population transfer.

    Supply exactly one of p, rate, and t_phi, by keyword. Simulators use p for
    a dimension-generic channel: it preserves every population and
    multiplies every off-diagonal matrix element by ``1 - p``. The probability
    must be a finite ``int`` or ``float`` other than ``bool`` in ``[0, 1]``.

    An emulator uses rate for a local Lindblad operator. The rate is a finite,
    nonnegative ``int`` or ``float`` other than ``bool``, in the inverse of the
    backend's time unit. In dimension d its Lindblad operator is
    ``sqrt(2 * rate) * diag(0, 1, ..., d - 1)``. Coherence between levels j and
    k therefore decays at ``rate * (j - k)**2``. A finite positive ``t_phi``
    value of the same numeric types is shorthand for ``rate = 1 / t_phi``;
    the object stores the resulting rate.

    Backends do not convert between simulator and emulator forms. The scalar
    conversion methods match the two-level channel and adjacent-level
    coherence. For more than two levels, the simulator form damps all
    coherences uniformly while the emulator form depends on level separation,
    so the forms are not generally equivalent. This noise acts on one
    subsystem.

    Args:
        p: Simulator full-dephasing probability.
        rate: Emulator Lindblad-operator rate.
        t_phi: Pure-dephasing time in the backend's time unit.

    Raises:
        ValueError: If not exactly one argument is supplied, p or rate is
            outside its accepted range, or t_phi is not finite and positive.

    Attributes:
        p: Provided probability, or ``None`` in emulator mode.
        rate: Provided or t_phi-derived rate, or ``None`` in probability mode.

    Examples:
        >>> import fatqat as fq
        >>> fq.noise.PhaseDamping(p=0.015).p
        0.015
        >>> fq.noise.PhaseDamping(t_phi=500.0).rate
        0.002
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
        """Return the scalar dephasing probability for a duration.

        Probability mode returns p unchanged after validating duration. Rate
        mode returns ``1 - exp(-rate * duration)``. This gives the equivalent
        two-level channel and adjacent-level coherence decay. It does not make
        the simulator and emulator multilevel conventions generally equivalent.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the same time unit whose inverse is used by rate.

        Returns:
            The provided probability or the probability converted from the
            emulator rate.

        Raises:
            ValueError: If duration is not a finite nonnegative real number.
        """
        if self.p is not None:
            _require_duration(duration)
            return self.p
        return _rate_to_p(self.rate, duration)

    def as_rate(self, duration: float) -> float:
        """Return the scalar dephasing rate for a duration.

        Rate mode returns rate unchanged after validating duration.
        Probability mode returns ``-log(1 - p) / duration``. This is the
        equivalent two-level or adjacent-coherence rate, not a general
        multilevel channel equivalence.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the time unit for the returned rate.

        Returns:
            The provided rate or the rate converted from the simulator
            probability.

        Raises:
            ValueError: If duration is invalid, p is 1, or a nonzero p is
                converted at zero duration.
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
