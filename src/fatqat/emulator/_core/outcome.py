"""Private representation-neutral values at the pulse backend boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

PulseMethod = Literal["statevector", "density_matrix", "unitary"]
ExecutionMode = Literal["density_matrix", "statevector", "trajectory"]
FinalStateKind = Literal["statevector", "density_matrix"]


@dataclass(frozen=True)
class _PulseResultRequest:
    counts: bool
    final_state: bool
    method: PulseMethod
    execution_mode: ExecutionMode | None


@dataclass(frozen=True)
class _PulseShotOutcome:
    """One shot in canonical physical-axis order at the backend boundary.

    A retained statevector is flat in canonical little-endian basis order. A
    retained density matrix uses that same order for both rows and columns.
    """

    final_state: np.ndarray | None
    final_state_kind: FinalStateKind
    classical_digits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PulseExecutionSummary:
    """Complete result-ready facts produced by one numerical execution."""

    outcomes: tuple[_PulseShotOutcome, ...]
    final_state_kind: FinalStateKind
    solver_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(
            self,
            "solver_metadata",
            MappingProxyType(dict(self.solver_metadata)),
        )
