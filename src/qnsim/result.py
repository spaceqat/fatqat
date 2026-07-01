"""Result and ResultConfig, plus the counts assembly helper."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import ResultFieldUnavailableError


@dataclass(frozen=True)
class ResultConfig:
    """Requested result fields for backend execution.

    `None` asks the backend to choose the phase-appropriate default. `True`
    explicitly requests a field, and `False` explicitly suppresses it.

    Attributes:
        counts: Whether to include sampled measurement counts.
        statevector: Whether to include a statevector snapshot.
    """

    counts: bool | None = None
    statevector: bool | None = None


def build_counts(
    indices: Iterable[int],
    n_clbits: int,
    measurements: Sequence[tuple[int, int]],
) -> dict[str, int]:
    """Build little-endian count keys from sampled basis-state indices.

    Measurement mappings are applied in program order, so later writes to the
    same classical bit replace earlier writes. Unwritten classical bits stay 0.

    Args:
        indices: Sampled flat basis-state indices.
        n_clbits: Number of classical bits in the result key.
        measurements: `(qubit_flat, clbit_flat)` pairs in program order.

    Returns:
        Count dictionary keyed by bitstrings with classical bit 0 rightmost.
    """
    counts: dict[str, int] = {}
    for idx in indices:
        idx = int(idx)
        clbits = [0] * n_clbits
        for q, c in measurements:
            clbits[c] = (idx >> q) & 1
        key = count_key_from_clbits(clbits, n_clbits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_key_from_clbits(clbits: Sequence[int], n_clbits: int) -> str:
    """Build a little-endian count key from one classical-register snapshot."""
    return "".join(str(clbits[c]) for c in range(n_clbits - 1, -1, -1))


def build_counts_from_clbits(
    snapshots: Iterable[Sequence[int]],
    n_clbits: int,
) -> dict[str, int]:
    """Aggregate per-shot classical-register snapshots into counts.

    Each snapshot is one shot's final classical-register state: a sequence of
    ``n_clbits`` bit values indexed by flat clbit index. Keys are little-endian
    bitstrings with classical bit 0 rightmost, matching ``build_counts``.

    Args:
        snapshots: One classical-register snapshot per shot.
        n_clbits: Number of classical bits in each key.

    Returns:
        Count dictionary keyed by little-endian classical bitstrings.
    """
    counts: dict[str, int] = {}
    for snap in snapshots:
        key = count_key_from_clbits(snap, n_clbits)
        counts[key] = counts.get(key, 0) + 1
    return counts


class Result:
    """Execution result with explicitly available data fields.

    Accessors raise `ResultFieldUnavailableError` when a field was not produced
    by the backend. Check `available_data` or use the accessor errors to handle
    optional fields. `metadata` always exists and stores backend/run context
    such as shots, backend name, and the effective result configuration.
    """

    def __init__(
        self,
        counts: dict[str, int] | None = None,
        statevector: np.ndarray | None = None,
        available: frozenset[str] = frozenset(),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a result from backend-produced fields.

        Args:
            counts: Optional measurement-count dictionary.
            statevector: Optional statevector array.
            available: Field names that accessors are allowed to return.
            metadata: Optional backend/run metadata copied into the public
                `metadata` dictionary.
        """
        self._counts = counts
        self._statevector = statevector
        self.available_data = frozenset(available)
        self.metadata = dict(metadata) if metadata is not None else {}

    def get_counts(self) -> dict[str, int]:
        """Return measurement counts.

        Returns:
            Count dictionary keyed by little-endian classical bitstrings.

        Raises:
            ResultFieldUnavailableError: If counts were not produced.

        Examples:
            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)
            result = qs.StateVectorBackend().run(program, shots=10).result()

            assert result.get_counts() == {"1": 10}
            ```
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
            ```python
            import qnsim as qs

            program = qs.Program(1)
            program.add(qs.ops.X, 0)
            result = qs.StateVectorBackend().run(
                program,
                result_config=qs.ResultConfig(counts=False, statevector=True),
            ).result()

            statevector = result.get_statevector()
            ```
        """
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector
