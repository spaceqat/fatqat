"""Observables: weighted sums of local Pauli and projector factors.

An `Observable` is a Hermitian operator written as a weighted sum of terms,
where each term is a tensor product of single-qubit factors. It is the input
to :py:class:`~fatqat.Estimator`, which reports ``<psi|O|psi>``.

The representation deliberately mirrors Qiskit's ``SparsePauliOp``/
``SparseObservable``: a term is described by *letters*, never by a matrix, and
the full ``2**n x 2**n`` operator is never built. Only the non-identity factors
of a term are stored, so memory grows with the number of factors actually
written, not with the qubit count - a two-body term on 100 qubits costs the
same as one on 4.

Labels are little-endian: the rightmost character is qubit 0, matching both
Qiskit's convention and fatqat's own counts strings (see
``result.format_count_key``), so users never switch conventions mid-analysis.

Supported letters:

===========  ==========================  ==================================
letter       operator                    role
===========  ==========================  ==================================
``I``        identity                    omitted from storage
``X``        Pauli X                     off-diagonal (permutes basis)
``Y``        Pauli Y                     off-diagonal, contributes a phase
``Z``        Pauli Z                     diagonal, contributes a sign
``ZERO``     ``|0><0|``                  diagonal projector
``ONE``      ``|1><1|``                  diagonal projector
===========  ==========================  ==================================

``ZERO``/``ONE`` make site occupation directly expressible: ``<ONE_i>`` is the
occupation number of qubit ``i``, which is the quantity atom-array experiments
report. The non-diagonal projectors (``PLUS``/``MINUS``/``RIGHT``/``LEFT``)
are not part of this version; the storage shape already accommodates them.

Observables are qubit-only: the letter alphabet has no meaning for ``dim > 2``
registers (see the design note in ``docs/design``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

# Single-character labels usable inside a dense label string. The two
# projectors have multi-character names, so they are only reachable through
# `from_sparse`, where each factor is named separately.
_DENSE_LETTERS = frozenset("IXYZ")
_ALL_LETTERS = frozenset({"I", "X", "Y", "Z", "ZERO", "ONE"})

# Letters whose local operator is diagonal. A term's diagonal factors never
# move amplitude between basis states, which is what lets the expectation
# kernels treat them as per-index weights rather than as a matrix product.
_DIAGONAL_LETTERS = frozenset({"Z", "ZERO", "ONE"})


class Observable:
    """A Hermitian sum of weighted Pauli/projector terms.

    Construct from dense labels, from label/coefficient sequences, or from a
    sparse factor list. All three build the same internal form: one entry per
    term holding its coefficient and only its non-identity factors.

    Examples:
        Three equivalent ways to write ``1.5 * ZZ``:

        >>> import fatqat as fq
        >>> a = fq.Observable([("ZZ", 1.5)])
        >>> b = fq.Observable(["ZZ"], coeffs=[1.5])
        >>> c = fq.Observable.from_sparse([("ZZ", (1, 0), 1.5)], num_qubits=2)
        >>> a == b == c
        True

        Site occupation on qubit 5 of a 100-qubit register, without writing a
        100-character label:

        >>> occupation = fq.Observable.from_sparse(
        ...     [("ONE", (5,), 1.0)], num_qubits=100
        ... )
        >>> occupation.num_qubits
        100
    """

    __slots__ = ("_terms", "_num_qubits")

    def __init__(
        self,
        data: Iterable[tuple[str, complex]] | Sequence[str],
        coeffs: Sequence[complex] | None = None,
        *,
        num_qubits: int | None = None,
    ) -> None:
        """Build an observable from dense labels.

        Args:
            data: Either ``[(label, coefficient), ...]`` or, when ``coeffs`` is
                given, a sequence of labels. Labels are little-endian strings
                over ``I``/``X``/``Y``/``Z`` and must all have the same length.
            coeffs: Coefficients paired with ``data`` when it holds bare
                labels. Must match ``data`` in length.
            num_qubits: Qubit count. Inferred from the label length when
                omitted; when given, it must agree with the labels.

        Raises:
            ValueError: If the term list is empty, labels disagree in length or
                with ``num_qubits``, a letter is not recognized, or any
                coefficient is not real (an observable must be Hermitian).
            TypeError: If ``data`` is not one of the two accepted shapes.
        """
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
            # Little-endian: position 0 from the right is qubit 0.
            for qubit, letter in enumerate(reversed(label)):
                if letter not in _DENSE_LETTERS:
                    raise ValueError(
                        f"unknown letter {letter!r} in label {label!r}; dense "
                        f"labels use {sorted(_DENSE_LETTERS)} (the ZERO/ONE "
                        "projectors are available through from_sparse)"
                    )
                if letter != "I":
                    factors.append((qubit, letter))
            terms.append((_require_real(coeff, label), tuple(factors)))

        self._terms = tuple(terms)
        self._num_qubits = width

    @classmethod
    def from_sparse(
        cls,
        data: Iterable[tuple[str | Sequence[str], Sequence[int], complex]],
        *,
        num_qubits: int,
    ) -> "Observable":
        """Build an observable by naming only the non-identity factors.

        This is the practical constructor for large registers, and the only way
        to use the ``ZERO``/``ONE`` projectors (whose names do not fit a
        single-character dense label).

        Args:
            data: ``[(letters, qubits, coefficient), ...]``. ``letters`` is
                either a string of single-character letters (``"XY"``) or a
                sequence of letter names (``["ONE", "Z"]``), paired positionally
                with ``qubits``.
            num_qubits: Total qubit count of the operator.

        Returns:
            The assembled observable.

        Raises:
            ValueError: If the term list is empty, ``letters`` and ``qubits``
                disagree in length, a qubit is out of range or repeats within a
                term, a letter is unknown, or a coefficient is not real.
        """
        num_qubits = _as_index(num_qubits, "num_qubits")
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

        terms = []
        for entry in data:
            letters, qubits, coeff = entry
            names = _split_letters(letters)
            qubits = list(qubits)
            if len(names) != len(qubits):
                raise ValueError(
                    f"letters {letters!r} and qubits {tuple(qubits)} must have "
                    "the same length"
                )
            factors = []
            seen: set[int] = set()
            for qubit, letter in zip(qubits, names):
                if letter not in _ALL_LETTERS:
                    raise ValueError(
                        f"unknown letter {letter!r}; expected one of "
                        f"{sorted(_ALL_LETTERS)}"
                    )
                qubit = _as_index(qubit, "qubit")
                if not 0 <= qubit < num_qubits:
                    raise ValueError(
                        f"qubit {qubit} is out of range for num_qubits={num_qubits}"
                    )
                if qubit in seen:
                    raise ValueError(
                        f"qubit {qubit} appears more than once in one term; each "
                        "term holds a single factor per qubit"
                    )
                seen.add(qubit)
                if letter != "I":
                    factors.append((qubit, letter))
            # Sorted so two spellings of the same term compare equal.
            terms.append((_require_real(coeff, str(letters)), tuple(sorted(factors))))

        if not terms:
            raise ValueError(
                "an observable needs at least one term; got an empty term list"
            )
        observable = cls.__new__(cls)
        observable._terms = tuple(terms)
        observable._num_qubits = num_qubits
        return observable

    @property
    def num_qubits(self) -> int:
        """Number of qubits this observable is defined on."""
        return self._num_qubits

    @property
    def terms(self) -> tuple[tuple[float, tuple[tuple[int, str], ...]], ...]:
        """The terms as ``(coefficient, ((qubit, letter), ...))`` tuples.

        Identity factors are absent, so an identity-only term carries an empty
        factor tuple. Factors are sorted by qubit.
        """
        return self._terms

    def is_diagonal(self) -> bool:
        """Whether every factor is diagonal (``Z``/``ZERO``/``ONE``).

        A diagonal observable can be evaluated from the basis-state
        probabilities alone, without touching amplitudes.
        """
        return all(
            letter in _DIAGONAL_LETTERS
            for _, factors in self._terms
            for _, letter in factors
        )

    def __len__(self) -> int:
        """Number of terms, including identity and zero-coefficient terms."""
        return len(self._terms)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Observable):
            return NotImplemented
        return self._num_qubits == other._num_qubits and self._terms == other._terms

    def __hash__(self) -> int:
        return hash((self._num_qubits, self._terms))

    def __repr__(self) -> str:
        rendered = " + ".join(
            f"{coeff:g}*"
            + ("I" if not factors else "".join(f"{l}{w}" for w, l in factors))
            for coeff, factors in self._terms
        )
        return f"<Observable on {self._num_qubits} qubits: {rendered}>"


def _as_index(value: Any, what: str) -> int:
    """Return ``value`` as a plain ``int``, accepting NumPy integers.

    Qubit indices are routinely produced by NumPy - ``rng.choice(n, 2)`` when
    sampling a lattice, ``np.arange`` when walking a chain - and those are
    ``np.int64``, not ``int``. Refusing them would reject the natural way to
    build a many-body observable, so they are accepted and narrowed here;
    Qiskit accepts them too.

    Booleans stay rejected, in both flavors. ``bool`` is a subclass of ``int``
    and ``np.bool_`` sits alongside ``np.integer``, so ``True`` would otherwise
    slip through as qubit 1 - a typo that silently means something.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {value!r}")
    return int(value)


def _split_letters(letters: str | Sequence[str]) -> list[str]:
    """Split a sparse term's letter specification into one name per qubit.

    A plain string is the common shorthand for single-character letters
    (``"XY"`` means X then Y), but ``ZERO``/``ONE`` are multi-character names
    that must not be split into characters. The two cases are distinguished by
    content, not by a flag: a string splits per character only when every
    character is itself a valid single-character letter, which ``"ONE"`` and
    ``"ZERO"`` are not. Mixed terms are written as an explicit sequence, e.g.
    ``(["ONE", "Z"], (5, 3), 1.0)``.
    """
    if not isinstance(letters, str):
        return list(letters)
    if letters and all(char in _DENSE_LETTERS for char in letters):
        return list(letters)
    return [letters]


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

    Every supported letter is Hermitian, so the whole observable is Hermitian
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
