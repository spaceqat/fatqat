"""Weighted sums of products of single-subsystem observable factors."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["NumberOperator", "Observable", "Projector"]


def _as_index(value: Any, what: str) -> int:
    """Return ``value`` as a plain ``int``, accepting NumPy integers.

    Subsystem indices are routinely produced by NumPy - ``rng.choice(n, 2)``
    when sampling a lattice, ``np.arange`` when walking a chain - and those are
    ``np.int64``, not ``int``. Refusing them would reject the natural way to
    build a many-body observable, so they are accepted and narrowed here;
    Qiskit accepts them too.

    Booleans stay rejected, in both flavors. ``bool`` is a subclass of ``int``
    and ``np.bool_`` sits alongside ``np.integer``, so ``True`` would otherwise
    slip through as index 1 - a typo that silently means something.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {value!r}")
    return int(value)


class _Factor:
    """One single-subsystem factor of an observable term.

    A factor names an operator without a subsystem dimension. The backend that
    evaluates the observable checks the factor against its target's dimension
    once that dimension is known.
    """

    __slots__ = ()

    @property
    def _is_diagonal(self) -> bool:
        """Whether the factor is diagonal in the computational basis."""
        return True


@dataclass(frozen=True, slots=True, repr=False)
class _Pauli(_Factor):
    """A Pauli ``X``, ``Y``, or ``Z`` factor, defined on qubits only."""

    letter: str

    @property
    def _is_diagonal(self) -> bool:
        return self.letter == "Z"

    def __str__(self) -> str:
        return self.letter

    def __repr__(self) -> str:
        return self.letter


@dataclass(frozen=True, slots=True)
class NumberOperator(_Factor):
    """Number operator ``n = sum_k k |k><k|`` on one subsystem.

    On an optical mode ``<n>`` is the mean photon number; on a qudit it is the
    mean level; on a qubit ``n`` equals ``Projector(1)``. The operator has no
    dimension of its own, so it applies to a subsystem of any dimension.

    Examples:
        >>> import fatqat as fq
        >>> from fatqat.observable import NumberOperator
        >>> fq.Observable.from_sparse([([NumberOperator(), NumberOperator()], (0, 1), 1.0)])
        <Observable: 1*N@0 N@1>
    """

    def __str__(self) -> str:
        return "N"


@dataclass(frozen=True, slots=True)
class Projector(_Factor):
    """Projector ``|level><level|`` onto one computational-basis level.

    ``Projector(0)`` and ``Projector(1)`` are the ``ZERO`` and ``ONE``
    projectors. A backend rejects a level that its target's dimension cannot
    hold.

    Args:
        level: Non-negative level. Python and NumPy integers are accepted;
            booleans are not.

    Raises:
        TypeError: If ``level`` is not an accepted integer.
        ValueError: If ``level`` is negative.
    """

    level: int

    def __post_init__(self) -> None:
        level = _as_index(self.level, "level")
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        object.__setattr__(self, "level", level)

    def __str__(self) -> str:
        return f"P{self.level}"


# Letter shorthands. Dense labels use only the single-character ones; ZERO and
# ONE have multi-character names and are only reachable through from_sparse.
_DENSE_LETTERS = frozenset("IXYZ")
_LETTER_FACTORS: dict[str, _Factor | None] = {
    "I": None,
    "X": _Pauli("X"),
    "Y": _Pauli("Y"),
    "Z": _Pauli("Z"),
    "ZERO": Projector(0),
    "ONE": Projector(1),
}

_Factors = tuple[tuple[int, _Factor], ...]


class Observable:
    """Represent a Hermitian observable as a sum of weighted factor products.

    Each term is a real coefficient times a product of single-subsystem
    factors: the Pauli letters ``X``, ``Y``, and ``Z``, :class:`Projector`
    levels (``ZERO`` and ``ONE`` are shorthand for levels 0 and 1), and the
    :class:`NumberOperator`. An observable has no width or subsystem
    dimensions; the backend checks every factor against the program it runs.
    Paulis need a qubit, a projector level must be below its subsystem's
    dimension, and every index must name a program subsystem. Unnamed
    subsystems carry the identity. Repeated and zero-coefficient terms are
    preserved.

    Examples:
        Three equivalent ways to write ``1.5 * ZZ``:

        >>> import fatqat as fq
        >>> a = fq.Observable([("ZZ", 1.5)])
        >>> b = fq.Observable(["ZZ"], coeffs=[1.5])
        >>> c = fq.Observable.from_sparse([("ZZ", (1, 0), 1.5)])
        >>> a == b == c
        True

    """

    __slots__ = ("_terms", "_num_qubits")

    def __init__(
        self,
        data: Iterable[tuple[str, complex]] | Sequence[str],
        coeffs: Sequence[complex] | None = None,
        *,
        num_qubits: int | None = None,
    ) -> None:
        """Build an observable from dense labels in public subsystem order.

        Args:
            data: Either ``[(label, coefficient), ...]`` or, when ``coeffs`` is
                supplied, an iterable of labels. Every label must be a string
                over ``I``, ``X``, ``Y``, and ``Z`` with the same length. The
                leftmost character acts on subsystem 0, and subsystems past
                the label carry the identity. Every coefficient must have zero
                imaginary part and is stored as a float.
            coeffs: Coefficients paired with bare labels in ``data``. The two
                iterables must have the same length.
            num_qubits: Deprecated; an observable no longer has a width. When
                given, it must equal the label length.

        Raises:
            TypeError: If ``data`` does not use one of the accepted forms.
            ValueError: If no terms are supplied; labels have different
                lengths; ``num_qubits`` disagrees with that length; a label has
                an unsupported letter; coefficient counts differ; or a
                coefficient has a nonzero imaginary part.
        """
        if num_qubits is not None:
            _warn_num_qubits_deprecated()
        pairs = _normalize_dense_input(data, coeffs)
        if not pairs:
            raise ValueError(
                "an observable needs at least one term; got an empty term list"
            )

        widths = {len(label) for label, _ in pairs}
        if len(widths) != 1:
            raise ValueError(
                f"all labels must have the same length, got lengths {sorted(widths)}"
            )
        width = widths.pop()
        if num_qubits is not None and num_qubits != width:
            raise ValueError(
                f"num_qubits={num_qubits} disagrees with label width {width}"
            )

        terms = []
        for label, coeff in pairs:
            factors = []
            # Dense labels follow public subsystem order: character q names q.
            for index, letter in enumerate(label):
                if letter not in _DENSE_LETTERS:
                    raise ValueError(
                        f"unknown letter {letter!r} in label {label!r}; dense "
                        f"labels use {sorted(_DENSE_LETTERS)} (projectors and "
                        "the number operator are available through from_sparse)"
                    )
                factor = _LETTER_FACTORS[letter]
                if factor is not None:
                    factors.append((index, factor))
            terms.append((_require_real(coeff, label), tuple(factors)))

        self._terms = tuple(terms)
        self._num_qubits = width

    @classmethod
    def from_sparse(
        cls,
        data: Iterable[
            tuple[str | _Factor | Sequence[str | _Factor], Sequence[int], complex]
        ],
        *,
        num_qubits: int | None = None,
    ) -> "Observable":
        """Build an observable by naming only the non-identity factors.

        Use this form for wide observables, for projectors, and for the number
        operator, none of which fit a dense single-character label.

        Args:
            data: Iterable of ``(factors, indices, coefficient)`` terms.
                ``factors`` may be an ``I``/``X``/``Y``/``Z`` string such as
                ``"XY"``; one factor name or object, such as ``"ONE"`` or
                ``NumberOperator()``; or a sequence mixing names and objects,
                such as ``["Z", Projector(2)]``. The names are ``I``, ``X``,
                ``Y``, ``Z``, ``ZERO`` (``Projector(0)``), and ``ONE``
                (``Projector(1)``). Factors pair with subsystem indices
                positionally. Each index may appear at most once in a term.
                Python and NumPy integers are accepted as indices; booleans
                are not.
            num_qubits: Deprecated; an observable no longer has a width. When
                given, it must be at least one and every index must be below
                it.

        Returns:
            The assembled observable.

        Raises:
            TypeError: If ``num_qubits`` or an index is not an accepted
                integer; ``factors`` or ``indices`` is not iterable; or a
                factor is neither a name nor a factor object.
            ValueError: If no terms are supplied; ``num_qubits`` is less than
                one; factors and indices have different lengths; an index is
                negative, at least ``num_qubits``, or repeated; a factor name
                is unknown; or a coefficient has a nonzero imaginary part.

        Examples:
            Name only the two factors in a wide observable:

            >>> import fatqat as fq
            >>> occupation = fq.Observable.from_sparse([(["ONE", "Z"], (5, 3), 1.0)])
            >>> occupation.terms
            ((1.0, ((3, Z), (5, Projector(level=1)))),)
        """
        if num_qubits is not None:
            _warn_num_qubits_deprecated()
            num_qubits = _as_index(num_qubits, "num_qubits")
            if num_qubits < 1:
                raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

        terms = []
        for entry in data:
            letters, indices, coeff = entry
            factors_in = _split_letters(letters)
            indices = list(indices)
            if len(factors_in) != len(indices):
                raise ValueError(
                    f"factors {letters!r} and indices {tuple(indices)} must have "
                    "the same length"
                )
            factors = []
            seen: set[int] = set()
            for index, letter in zip(indices, factors_in):
                factor = _as_factor(letter)
                index = _as_index(index, "index")
                if index < 0:
                    raise ValueError(f"index {index} is out of range; must be >= 0")
                if num_qubits is not None and index >= num_qubits:
                    raise ValueError(
                        f"index {index} is out of range for num_qubits={num_qubits}"
                    )
                if index in seen:
                    raise ValueError(
                        f"index {index} appears more than once in one term; each "
                        "term holds a single factor per subsystem"
                    )
                seen.add(index)
                if factor is not None:
                    factors.append((index, factor))
            # Sorted so two spellings of the same term compare equal.
            terms.append(
                (
                    _require_real(coeff, str(letters)),
                    tuple(sorted(factors, key=lambda item: item[0])),
                )
            )

        if not terms:
            raise ValueError(
                "an observable needs at least one term; got an empty term list"
            )
        observable = cls.__new__(cls)
        observable._terms = tuple(terms)
        if num_qubits is None:
            num_qubits = 1 + max(
                (index for _, factors in terms for index, _ in factors), default=-1
            )
        observable._num_qubits = num_qubits
        return observable

    @property
    def num_qubits(self) -> int:
        """Deprecated width of this observable.

        Returns the ``num_qubits`` given at construction, the dense label
        length, or one more than the largest named index.
        """
        _warn_num_qubits_deprecated()
        return self._num_qubits

    @property
    def terms(self) -> tuple[tuple[float, _Factors], ...]:
        """Return normalized ``(coefficient, factors)`` tuples.

        Identity factors are absent, so an identity-only term carries an empty
        factor tuple. Each factor is an ``(index, factor)`` pair, sorted by
        index. Pauli factors print as their letter; ``ZERO`` and ``ONE`` are
        stored as ``Projector(0)`` and ``Projector(1)``. The returned
        structure is immutable.
        """
        return self._terms

    def is_diagonal(self) -> bool:
        """Return whether every factor is ``Z``, a projector, or ``N``.

        Identity-only terms are diagonal because identity factors are omitted.
        """
        return all(
            factor._is_diagonal for _, factors in self._terms for _, factor in factors
        )

    def __len__(self) -> int:
        """Number of terms, including identity and zero-coefficient terms."""
        return len(self._terms)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Observable):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(self._terms)

    def __repr__(self) -> str:
        rendered = " + ".join(
            f"{coeff:g}*"
            + (
                "I"
                if not factors
                else " ".join(f"{factor}@{index}" for index, factor in factors)
            )
            for coeff, factors in self._terms
        )
        return f"<Observable: {rendered}>"


def _warn_num_qubits_deprecated() -> None:
    warnings.warn(
        "num_qubits is deprecated: an Observable no longer has a width. The "
        "backend checks factor indices against the program it runs",
        DeprecationWarning,
        stacklevel=3,
    )


def _split_letters(
    letters: str | _Factor | Sequence[str | _Factor],
) -> list[str | _Factor]:
    """Split a sparse term's factor specification into one entry per index.

    A plain string is the common shorthand for single-character letters
    (``"XY"`` means X then Y), but ``ZERO``/``ONE`` are multi-character names
    that must not be split into characters. The two cases are distinguished by
    content, not by a flag: a string splits per character only when every
    character is itself a valid single-character letter, which ``"ONE"`` and
    ``"ZERO"`` are not. A lone factor object names one index. Mixed terms are
    written as an explicit sequence, e.g. ``(["ONE", "Z"], (5, 3), 1.0)``.
    """
    if isinstance(letters, _Factor):
        return [letters]
    if not isinstance(letters, str):
        return list(letters)
    if letters and all(char in _DENSE_LETTERS for char in letters):
        return list(letters)
    return [letters]


def _as_factor(letter: str | _Factor) -> _Factor | None:
    """Return the factor a name or object stands for; ``None`` for identity."""
    if isinstance(letter, _Factor):
        return letter
    if not isinstance(letter, str):
        raise TypeError(
            f"a factor must be a name or a factor object such as NumberOperator() "
            f"or Projector(level), got {letter!r}"
        )
    if letter not in _LETTER_FACTORS:
        raise ValueError(
            f"unknown letter {letter!r}; expected one of "
            f"{sorted(_LETTER_FACTORS)}, or a factor object such as "
            "NumberOperator() or Projector(level)"
        )
    return _LETTER_FACTORS[letter]


def _normalize_dense_input(
    data: Iterable[tuple[str, complex]] | Sequence[str],
    coeffs: Sequence[complex] | None,
) -> list[tuple[str, complex]]:
    """Reduce both accepted dense shapes to a list of ``(label, coeff)``."""
    if coeffs is not None:
        labels = list(data)
        coefficients = list(coeffs)
        if len(labels) != len(coefficients):
            raise ValueError(
                f"got {len(labels)} label(s) but {len(coefficients)} coefficient(s)"
            )
        for label in labels:
            if not isinstance(label, str):
                raise TypeError(
                    f"with coeffs= given, data must hold label strings, got {label!r}"
                )
        return list(zip(labels, coefficients))

    pairs = []
    for entry in data:
        if isinstance(entry, (str, Mapping)):
            raise TypeError(
                "without coeffs=, data must hold (label, coefficient) pairs; "
                f"got {entry!r}"
            )
        label, coeff = entry
        if not isinstance(label, str):
            raise TypeError(f"label must be a string, got {label!r}")
        pairs.append((label, coeff))
    return pairs


def _require_real(coeff: complex, label: str) -> float:
    """Return ``coeff`` as a float, rejecting a complex coefficient.

    Every supported factor is Hermitian, so the whole observable is Hermitian
    exactly when its coefficients are real - and then every expectation value
    is real too. Rejecting here, at construction, keeps that guarantee at the
    one place it is cheap to state.
    """
    value = complex(coeff)
    if value.imag != 0.0:
        raise ValueError(
            f"coefficient {coeff!r} for term {label!r} is complex; an "
            "observable must be Hermitian, so coefficients must be real"
        )
    return value.real
