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
    """One shot in the owning pulse allocation's order at the backend boundary.

    Built-in QuTiP adapters use public model/factor order. A retained density
    matrix uses the same order for both rows and columns.
    """

    final_state: np.ndarray | None
    final_state_kind: FinalStateKind
    classical_digits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PulseExecutionSummary:
    """Complete result-ready facts produced by one numerical execution."""

    outcomes: tuple[_PulseShotOutcome, ...]
    final_state_kind: FinalStateKind
    runtime_details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(
            self,
            "runtime_details",
            MappingProxyType(dict(self.runtime_details)),
        )
