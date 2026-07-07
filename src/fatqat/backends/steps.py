"""Resolved execution-plan step types shared by the statevector backend and engine.

A `Program` lowers to a `list[ResolvedStep]`: each step is either an
`ApplyMatrixStep` (a local matrix payload built from a matrix-implementation
rule plus layout-resolved target indices), a `MeasurementStep`, or a
`ResetStep`. Defined here, separate from both `implementation/` (the rule
protocol that only ever produces a bare matrix, never a step) and
`statevector.py`/`statevectorengine.py` (which would otherwise need to import
this type from each other and cycle), so both the backend and the engine can
import it without depending on one another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApplyMatrixStep:
    """Resolved local matrix payload consumed by the statevector engine.

    Doubles as the "apply a matrix" entry in a backend execution plan and as the
    payload the engine applies. The matrix is marked read-only after construction
    so this frozen value object cannot be mutated through the NumPy array buffer.

    Attributes:
        matrix: Local operation matrix.
        target_indices: Flat subsystem indices the matrix acts on.
        condition: Optional feedforward guard as lowered ``(clbit_index, value)``
            AND-terms. ``None`` means unconditional. The engine ignores this
            field; the backend's per-shot loop evaluates it.
    """

    matrix: np.ndarray
    target_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        # The engine consumes the matrix read-only; lock it so this frozen
        # dataclass is truly immutable (Python cannot freeze array contents).
        # A rule may hand back a shared/cached array (a `FixedMatrix` copies,
        # but a bare callable need not), so copy before freezing when the array
        # is still writeable - freezing in place would mutate the rule's own
        # object as a side effect. An already read-only array is left as-is.
        if self.matrix.flags.writeable:
            object.__setattr__(self, "matrix", np.array(self.matrix, copy=True))
        self.matrix.flags.writeable = False


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat subsystem indices into matching flat clbit indices."""

    measured_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]


@dataclass(frozen=True)
class ResetStep:
    """Resolved reset of one or more flat subsystems to |0>, with optional condition.

    `Reset` is an `AppliedOperation`, so it can carry a feedforward `condition`
    just like a gate. The lowered form stores it as ``(clbit_index, value)``
    AND-terms; the per-shot loop skips the reset when the guard fails.
    """

    reset_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None


ResolvedStep = ApplyMatrixStep | MeasurementStep | ResetStep
