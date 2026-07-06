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


def build_counts(
    indices: Iterable[int],
    n_clbits: int,
    measurements: Sequence[tuple[int, int]],
    system_dims: Sequence[int],
) -> dict[tuple[int, ...], int]:
    """Decode sampled flat basis indices into tuple-keyed counts.

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
        Count dictionary keyed by ascending flat clbit index (clbit 0 first).
    """
    strides = _radix_strides(system_dims)
    counts: dict[tuple[int, ...], int] = {}
    for idx in indices:
        idx = int(idx)
        clbits = [0] * n_clbits
        for q, c in measurements:
            clbits[c] = (idx // strides[q]) % system_dims[q]
        key = tuple(clbits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_counts_from_clbits(
    snapshots: Iterable[Sequence[int]],
    n_clbits: int,
) -> dict[tuple[int, ...], int]:
    """Aggregate per-shot classical-register snapshots into tuple-keyed counts.

    Each snapshot is one shot's final classical-register state: a sequence of
    ``n_clbits`` digit values indexed by flat clbit index, matching
    ``build_counts``.

    Args:
        snapshots: One classical-register snapshot per shot.
        n_clbits: Number of classical bits in each key.

    Returns:
        Count dictionary keyed by ascending flat clbit index (clbit 0 first).
    """
    counts: dict[tuple[int, ...], int] = {}
    for snap in snapshots:
        key = tuple(snap[c] for c in range(n_clbits))
        counts[key] = counts.get(key, 0) + 1
    return counts


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
        """
        self._counts = counts
        self._statevector = statevector
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
            .. code-block:: python

                import fatqcat as fqc

                program = fqc.Program(1, 1)
                program.add(fqc.ops.X, 0)
                program.add_measurement(0, 0)
                result = fqc.backends.StateVectorBackend().run(program, shots=10).result()

                assert result.get_counts() == {"1": 10}
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
            .. code-block:: python

                import fatqcat as fqc

                program = fqc.Program(1)
                program.add(fqc.ops.X, 0)
                result = fqc.backends.StateVectorBackend().run(
                    program,
                    result_config={"counts": False, "statevector": True},
                ).result()

                statevector = result.get_statevector()
        """
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector
