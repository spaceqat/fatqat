"""Result objects and counts assembly helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import ResultFieldUnavailableError


@dataclass(frozen=True)
class _ResultConfig:
    """Internal normalized result-field selection for one backend execution.

    Each field is tri-state:

    - `None`: let the backend choose the default for the given program
    - `True`: explicitly request the field
    - `False`: explicitly suppress the field

    For `StateVectorBackend`, the defaults are:

    - `counts=None`: produce counts when the program contains at least one
      measurement
    - `statevector=None`: produce a statevector only when execution is
      non-stochastic, meaning the program contains no measurement and no reset

    Explicit requests can further constrain `shots`. In particular, requesting
    `statevector=True` for a stochastic program (one with measurement or reset)
    is only supported for `shots == 1`.

    This helper is backend-internal. User-facing APIs accept plain dictionaries
    such as `{"counts": True, "statevector": False}` and normalize them into
    this shape before execution.
    """

    counts: bool | None = None
    statevector: bool | None = None


@dataclass(frozen=True)
class _DensityMatrixResultConfig:
    """Internal normalized result-field selection for one density-matrix execution.

    Each field is tri-state with the same meaning as `_ResultConfig`:
    `None` lets the backend choose, `True` requests, `False` suppresses.

    For `DensityMatrixBackend`, the defaults are:

    - `counts=None`: produce counts when the program contains at least one
      measurement
    - `density_matrix=None`: produce a density matrix only when the program
      contains no measurement. Reset does not suppress the default: on a
      density matrix, reset is a deterministic channel (partial trace plus
      repreparation), so a reset-bearing program still has a well-defined
      ensemble state.

    Requesting `density_matrix=True` for a program with measurement is only
    supported for `shots == 1`; the exported state is that single trajectory's
    post-measurement density matrix.
    """

    counts: bool | None = None
    density_matrix: bool | None = None


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
        """
        self._counts = counts
        self._statevector = statevector
        self._density_matrix = density_matrix
        self.available_data = frozenset(available)
        self.metadata = dict(metadata) if metadata is not None else {}
        self._classical_dims = tuple(classical_dims)

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
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.X, 0)
            >>> program.add_measurement(0, 0)
            >>> result = fq.backends.StateVectorBackend().run(program, shots=10).result()
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
            >>> program = fq.Program(1)
            >>> program.add(fq.ops.X, 0)
            >>> result = fq.backends.StateVectorBackend().run(
            ...     program,
            ...     result_config={"counts": False, "statevector": True},
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
            >>> program = fq.Program(1)
            >>> program.add(fq.ops.H, 0)
            >>> result = fq.backends.DensityMatrixBackend().run(
            ...     program,
            ...     result_config={"counts": False, "density_matrix": True},
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
