"""Result values returned by FATQAT execution APIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import BackendValidationError, ResultFieldUnavailableError


@dataclass(frozen=True)
class _ResultConfig:
    """Internal normalized request for data returned by one execution.

    ``counts`` is a hardware-readable result artifact. ``final_state`` is a
    simulator-only result field: its concrete representation is selected by
    the backend's ``method``. Hardware backends can derive a more specific
    config dataclass to expose additional result artifacts; a subclass that
    adds fields must validate those fields in its ``__post_init__``.
    """

    counts: bool | None = None
    final_state: bool | None = None

    def __post_init__(self) -> None:
        for name, value in (("counts", self.counts), ("final_state", self.final_state)):
            if value is not None and type(value) is not bool:
                raise BackendValidationError(
                    f"{name} must be bool or None, got {value!r}"
                )


def decode_indices_to_clbit_rows(
    indices: Iterable[int],
    measurements: Sequence[tuple[int, int]],
    system_dims: Sequence[int],
    n_clbits: int,
) -> np.ndarray:
    """Decode sampled flat basis indices into per-shot clbit rows.

    Each measured subsystem's digit is extracted by its own radix
    ``system_dims[qubit_flat]`` via little-endian place value. Later writes to
    the same clbit replace earlier ones; unwritten clbits stay 0.

    Args:
        indices: Sampled flat basis-state indices.
        n_clbits: Number of classical bits in the result key.
        measurements: `(qubit_flat, clbit_flat)` pairs in program order.
        system_dims: Per-subsystem dimensions of the quantum register, used to
            decode each measured subsystem's digit from the flat index.

    Returns:
        One row per shot, with clbit 0 in column 0.
    """
    index_array = np.asarray(list(indices), dtype=int)
    rows = np.zeros((len(index_array), n_clbits), dtype=int)
    if len(index_array) == 0:
        return rows

    strides = _radix_strides(system_dims)
    for q, c in measurements:
        rows[:, c] = (index_array // strides[q]) % system_dims[q]
    return rows


def reduce_to_counts(
    rows: Iterable[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-shot clbit rows to distinct rows and counts."""
    row_array = np.asarray(rows, dtype=int)
    if row_array.ndim == 1:
        row_array = (
            row_array.reshape((0, 0))
            if row_array.size == 0
            else row_array.reshape((1, -1))
        )
    if row_array.shape[0] == 0:
        return row_array, np.zeros(0, dtype=int)
    return np.unique(row_array, axis=0, return_counts=True)


def counts_dict_from_arrays(
    outcome_keys: np.ndarray,
    outcome_counts: np.ndarray,
) -> dict[tuple[int, ...], int]:
    """Package engine count arrays into the public tuple-keyed dict shape."""
    return {
        tuple(int(k) for k in row): int(count)
        for row, count in zip(outcome_keys, outcome_counts)
    }


def _radix_strides(dims: Sequence[int]) -> list[int]:
    """Return the mixed-radix place value (stride) for each subsystem."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def format_count_key(key: tuple[int, ...], classical_dims: Sequence[int]) -> str:
    """Render a tuple-keyed count as a little-endian display string.

    Single-character concatenation when every classical register dim is
    <= 9; a comma-delimited little-endian form once any classical dim is
    >= 10. The trigger is `classical_dims` (never quantum `system_dims`).

    Args:
        key: Tuple-keyed count in ascending flat clbit index order.
        classical_dims: Per-clbit classical dimensions, used only to decide
            the rendering form.

    Returns:
        A little-endian string (highest clbit first).
    """
    order = range(len(key) - 1, -1, -1)
    if all(d <= 9 for d in classical_dims):
        return "".join(str(key[c]) for c in order)
    return ",".join(str(key[c]) for c in order)


class Result:
    """Expose the data produced by one backend or estimator run.

    ``available_data`` is the immutable set of fields produced by this run.
    An unavailable accessor raises ``ResultFieldUnavailableError`` instead of
    returning ``None``. ``metadata`` is a mutable dictionary whose keys and
    values depend on the producer.

    Treat values returned by accessors as read-only. ``get_counts`` is the
    exception: it returns a new dictionary of formatted string keys.
    """

    def __init__(
        self,
        counts: dict[tuple[int, ...], int] | None = None,
        statevector: np.ndarray | None = None,
        available: frozenset[str] = frozenset(),
        metadata: Mapping[str, Any] | None = None,
        classical_dims: Sequence[int] = (),
        density_matrix: np.ndarray | None = None,
        unitary: np.ndarray | None = None,
        superop: np.ndarray | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a result from produced fields.

        Most users receive a ``Result`` from ``Job.result()``. ``metadata`` and
        ``data`` are shallow-copied; count dictionaries and arrays are retained
        by reference.

        Args:
            counts: Counts keyed by tuples in ascending flat classical-slot
                order.
            statevector: Statevector field, if produced.
            available: Core field names that their accessors may return.
            metadata: Producer-specific run metadata.
            classical_dims: Per-slot classical dimensions used to format count
                strings.
            density_matrix: Density-matrix field, if produced.
            unitary: Unitary field, if produced.
            superop: Super-operator field, if produced.
            data: Additional named fields. These names are added to
                ``available_data`` automatically and are read with
                ``get_data`` or a dedicated accessor.

        Raises:
            BackendValidationError: If ``data`` uses ``"counts"``,
                ``"final_state"``, ``"statevector"``, ``"density_matrix"``,
                ``"unitary"``, or ``"superop"``, which are reserved for core
                fields.
        """
        self._data = dict(data) if data is not None else {}
        reserved = {
            "counts",
            "final_state",
            "statevector",
            "density_matrix",
            "unitary",
            "superop",
        }
        collisions = reserved & self._data.keys()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise BackendValidationError(
                f"backend-specific data cannot replace reserved field(s) {names}"
            )
        self._counts = counts
        self._statevector = statevector
        self._density_matrix = density_matrix
        self._unitary = unitary
        self._superop = superop
        self.available_data = frozenset(available) | frozenset(self._data)
        self.metadata = dict(metadata) if metadata is not None else {}
        self._classical_dims = tuple(classical_dims)

    def get_data(self, name: str) -> Any:
        """Return an additional result field by name.

        Core fields such as counts and statevectors use their dedicated
        accessors.

        Args:
            name: Additional backend-specific field name.

        Returns:
            The field value.

        Raises:
            ResultFieldUnavailableError: If ``name`` is not an additional
                field in this result.
        """
        if name not in self._data:
            raise ResultFieldUnavailableError(f"{name} not available in this result")
        return self._data[name]

    def get_expectation(self) -> Any:
        """Return expectation values produced by an estimator.

        One observable produces a scalar. A list or tuple produces a
        one-dimensional NumPy array in input order.

        Raises:
            ResultFieldUnavailableError: If this result did not come from an
                estimator run.
        """
        if "expectation" not in self._data:
            raise ResultFieldUnavailableError(
                "expectation not available in this result; it is produced by "
                "Estimator.run, not by a backend run"
            )
        return self._data["expectation"]

    def get_std(self) -> Any:
        """Return estimator standard errors.

        The shape matches ``get_expectation``. An exact run with ``shots=0``
        reports zero. For a sampled run, this is the statistical standard
        error reported by the estimator, not a sample standard deviation.

        Raises:
            ResultFieldUnavailableError: If this result did not come from an
                estimator run.
        """
        if "std" not in self._data:
            raise ResultFieldUnavailableError(
                "std not available in this result; it is produced by "
                "Estimator.run, not by a backend run"
            )
        return self._data["std"]

    def get_counts(self) -> dict[str, int]:
        """Return a new counts dictionary with display-string keys.

        The highest-index classical slot appears on the left and slot 0 on
        the right. Digits are concatenated when every classical dimension is
        at most 9; otherwise commas separate them.

        Returns:
            A newly created dictionary; changing it does not change this
            result.

        Raises:
            ResultFieldUnavailableError: If counts were not produced.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> program = fq.Program(1, 1)
            >>> program.add(ops.X, 0)
            >>> program.measure(0, 0)
            >>> result = fq.simulator.Simulator("SV").run(program, shots=10).result()
            >>> result.get_counts()
            {'1': 10}
        """
        if "counts" not in self.available_data:
            raise ResultFieldUnavailableError("counts not available in this result")
        return {
            format_count_key(key, self._classical_dims): n
            for key, n in self._counts.items()
        }

    def get_counts_as_tuples(self) -> dict[tuple[int, ...], int]:
        """Return counts keyed by ascending flat classical-slot index.

        Returns:
            A dictionary whose tuple position 0 is classical slot 0.

        Raises:
            ResultFieldUnavailableError: If counts were not produced.
        """
        if "counts" not in self.available_data:
            raise ResultFieldUnavailableError("counts not available in this result")
        return self._counts

    def get_statevector(self) -> np.ndarray:
        """Return the produced statevector.

        Built-in backends return a length-``D`` array, where ``D`` is the
        product of subsystem dimensions. ``metadata["state_axes"]`` describes
        the least-significant-first subsystem order.

        Returns:
            The statevector array.

        Raises:
            ResultFieldUnavailableError: If a statevector was not produced.

        """
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector

    def get_density_matrix(self) -> np.ndarray:
        """Return the produced density matrix.

        Built-in backends return a ``(D, D)`` array, where ``D`` is the product
        of subsystem dimensions. ``metadata["state_axes"]`` describes the
        least-significant-first subsystem order used by both axes.

        Returns:
            The density-matrix array.

        Raises:
            ResultFieldUnavailableError: If a density matrix was not produced.

        """
        if "density_matrix" not in self.available_data:
            raise ResultFieldUnavailableError(
                "density_matrix not available in this result"
            )
        return self._density_matrix

    def get_unitary(self) -> np.ndarray:
        """Return the program's unitary matrix.

        Returns:
            A ``(D, D)`` array, where ``D`` is the product of the subsystem
            dimensions. Column ``j`` is the state the program prepares from
            basis state ``|j>``. Column 0 matches a default-state statevector
            run only when that backend uses the same terminal-frame convention
            for states and operators.

        Raises:
            ResultFieldUnavailableError: If a unitary was not produced.

        """
        if "unitary" not in self.available_data:
            raise ResultFieldUnavailableError("unitary not available in this result")
        return self._unitary

    def get_superop(self) -> np.ndarray:
        """Return the program's super-operator matrix.

        The representation uses column stacking: flatten both input and output
        density matrices with ``reshape(-1, order="F")``. A unitary program's
        super-operator is ``numpy.kron(U.conj(), U)``.

        Returns:
            A ``(D**2, D**2)`` array, where ``D`` is the product of the
            subsystem dimensions.

        Raises:
            ResultFieldUnavailableError: If a super-operator was not produced.

        """
        if "superop" not in self.available_data:
            raise ResultFieldUnavailableError("superop not available in this result")
        return self._superop
