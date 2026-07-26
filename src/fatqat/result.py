"""Result objects and counts assembly helpers."""

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
    """Execution result with explicitly available data fields.

    Accessors raise ``ResultFieldUnavailableError`` when a field was not
    produced by the backend. Check ``available_data`` or use the accessor
    errors to handle optional fields. ``metadata`` always exists and stores
    backend/run context such as shots, backend name, and the effective result
    configuration.
    """

    def __init__(
        self,
        counts: dict[tuple[int, ...], int] | None = None,
        statevector: np.ndarray | None = None,
        available: frozenset[str] = frozenset(),
        metadata: Mapping[str, Any] | None = None,
        classical_dims: Sequence[int] = (),
        density_matrix: np.ndarray | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a result from backend-produced fields.

        Args:
            counts: Optional measurement-count dictionary, keyed by ascending
                flat clbit index (clbit 0 first).
            statevector: Optional statevector array.
            available: Field names that accessors are allowed to return.
            metadata: Optional backend/run metadata copied into the public
                `metadata` dictionary.
            classical_dims: Per-clbit classical dimensions, used to render
                `get_counts()` display strings.
            density_matrix: Optional density-matrix array.
            data: Additional backend-specific result artifacts. Their names
                are included in :attr:`available_data` and retrieved with
                :meth:`get_data`.
        """
        self._data = dict(data) if data is not None else {}
        reserved = {"counts", "final_state", "statevector", "density_matrix"}
        collisions = reserved & self._data.keys()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise BackendValidationError(
                f"backend-specific data cannot replace reserved field(s) {names}"
            )
        self._counts = counts
        self._statevector = statevector
        self._density_matrix = density_matrix
        self.available_data = frozenset(available) | frozenset(self._data)
        self.metadata = dict(metadata) if metadata is not None else {}
        self._classical_dims = tuple(classical_dims)

    def get_data(self, name: str) -> Any:
        """Return a backend-specific result artifact by name.

        Raises:
            ResultFieldUnavailableError: If ``name`` was not produced by this
                backend run.
        """
        if name not in self._data:
            raise ResultFieldUnavailableError(f"{name} not available in this result")
        return self._data[name]

    def get_counts(self) -> dict[str, int]:
        """Return measurement counts as little-endian display strings.

        Returns:
            Count dictionary keyed by little-endian classical bitstrings
            (single-character digits when every classical dim is <= 9,
            otherwise a comma-delimited little-endian form).

        Raises:
            ResultFieldUnavailableError: If counts were not produced.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as op
            >>> program = fq.Program(1, 1)
            >>> program.add(op.X, 0)
            >>> program.add_measurement(0, 0)
            >>> result = fq.backends.SimulatorBackend("SV").run(program, shots=10).result()
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
        """Return measurement counts keyed by ascending flat clbit index.

        Returns:
            Count dictionary keyed by tuples with clbit 0 first, the same
            representation the backend produces internally.

        Raises:
            ResultFieldUnavailableError: If counts were not produced.
        """
        if "counts" not in self.available_data:
            raise ResultFieldUnavailableError("counts not available in this result")
        return self._counts

    def get_statevector(self) -> np.ndarray:
        """Return the statevector snapshot.

        Returns:
            A statevector array produced by the backend.

        Raises:
            ResultFieldUnavailableError: If a statevector was not produced.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as op
            >>> program = fq.Program(1)
            >>> program.add(op.X, 0)
            >>> result = fq.backends.SimulatorBackend("SV").run(
            ...     program,
            ...     result_config={"counts": False, "final_state": True},
            ... ).result()
            >>> result.get_statevector()
            array([0.+0.j, 1.+0.j])
        """
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector

    def get_density_matrix(self) -> np.ndarray:
        """Return the density-matrix snapshot.

        Returns:
            A density-matrix array produced by the backend.

        Raises:
            ResultFieldUnavailableError: If a density matrix was not produced.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as op
            >>> program = fq.Program(1)
            >>> program.add(op.H, 0)
            >>> result = fq.backends.SimulatorBackend("DM").run(
            ...     program,
            ...     result_config={"counts": False, "final_state": True},
            ... ).result()
            >>> result.get_density_matrix()
            array([[0.5+0.j, 0.5+0.j],
                   [0.5+0.j, 0.5+0.j]])
        """
        if "density_matrix" not in self.available_data:
            raise ResultFieldUnavailableError(
                "density_matrix not available in this result"
            )
        return self._density_matrix
