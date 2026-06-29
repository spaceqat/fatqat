"""Result and ResultConfig, plus the counts assembly helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ResultFieldUnavailableError


@dataclass(frozen=True)
class ResultConfig:
    counts: bool | None = None
    statevector: bool | None = None


def build_counts(indices, n_clbits, measurements) -> dict[str, int]:
    """Tally counts from sampled basis indices.

    measurements: list of (qubit_flat, clbit_flat) in program order; later writes
    to the same clbit override earlier ones. Key is little-endian (clbit 0 rightmost),
    unwritten clbits stay 0.
    """
    counts: dict[str, int] = {}
    for idx in indices:
        idx = int(idx)
        clbits = [0] * n_clbits
        for q, c in measurements:
            clbits[c] = (idx >> q) & 1
        key = "".join(str(clbits[c]) for c in range(n_clbits - 1, -1, -1))
        counts[key] = counts.get(key, 0) + 1
    return counts


class Result:
    def __init__(self, counts=None, statevector=None, available=frozenset()):
        self._counts = counts
        self._statevector = statevector
        self.available_data = frozenset(available)

    def get_counts(self) -> dict[str, int]:
        if "counts" not in self.available_data:
            raise ResultFieldUnavailableError("counts not available in this result")
        return self._counts

    def get_statevector(self) -> np.ndarray:
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector
