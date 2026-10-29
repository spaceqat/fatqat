"""Execution information for custom operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionContext:
    """Execution information for one custom-operation call.

    Attributes:
        targets: Quantum subsystem indices in operand order. These index
            ``system_dims`` and identify entries in ``state.classical.occupied``.
        system_dims: All subsystem dimensions, least-significant factor first.
        rng: This shot's random generator, controlled by the simulation seed.
    """

    targets: tuple[int, ...]
    system_dims: tuple[int, ...]
    rng: np.random.Generator
