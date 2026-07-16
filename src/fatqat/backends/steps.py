"""Resolved execution-plan step types shared by the simulator backend and simulators.

A :py:class:`~fatqat.Program` lowers to a `list[ResolvedStep]`: each step is either an
`ApplyMatrixStep` (a local matrix payload built from a matrix-implementation
rule plus layout-resolved target indices), an `ApplyChannelStep` (a Kraus
payload built from a channel-implementation rule, for channel-representable
noise), a `MeasurementStep`, or a `ResetStep`. Defined here, separate from
both `implementation/` / `noise/` (the rule protocols that only ever produce
bare arrays, never a step) and `simulator_backend.py` / the `simulator/`
package (which would otherwise need to import this type from each other and
cycle), so both the backend and the simulators can import it without
depending on one another.
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
class ApplyChannelStep:
    """Resolved channel payload: Kraus operators plus flat target indices.

    Built at lowering, right after the `ApplyMatrixStep` of the gate the
    channel is attached to (see `docs/design/architecture/noise-model/
    matrix-channel-noise.md` §4.3-4.4). Carries concrete arrays only - no
    descriptor or model handle survives into the engine or across a
    parallel-execution boundary. Each Kraus array is marked read-only after
    construction, same as `ApplyMatrixStep.matrix`.

    Attributes:
        kraus_ops: Resolved Kraus operators, CPTP-validated at lowering.
        target_indices: Flat subsystem indices the channel acts on.
        condition: The parent gate's lowered feedforward guard. A channel
            models its gate's noise, so when the guard skips the gate it
            skips the channel too. The engine ignores this field; the
            backend's per-shot loop evaluates it.
    """

    kraus_ops: tuple[np.ndarray, ...]
    target_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        # Same freezing policy as ApplyMatrixStep: copy any still-writeable
        # array (a rule may hand back a shared/cached one) before locking it.
        frozen = []
        for kraus in self.kraus_ops:
            if kraus.flags.writeable:
                kraus = np.array(kraus, copy=True)
            kraus.flags.writeable = False
            frozen.append(kraus)
        object.__setattr__(self, "kraus_ops", tuple(frozen))


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat subsystem indices into matching flat clbit indices.

    ``confusions`` carries classical readout error, resolved from the noise
    model at lowering: one optional column-stochastic confusion matrix per
    measured subsystem (aligned with ``measured_indices``), or ``None`` when
    no readout error applies to this measurement at all. The physical
    collapse always uses the true outcome; only the value written to the
    classical register is resampled through the matrix, so state export and
    qubit reuse are untouched and execution-strategy classification never
    changes.
    """

    measured_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]
    confusions: tuple[np.ndarray | None, ...] | None = None

    def __post_init__(self) -> None:
        # Same freezing policy as the array-carrying steps above.
        if self.confusions is None:
            return
        frozen: list[np.ndarray | None] = []
        for confusion in self.confusions:
            if confusion is not None:
                if confusion.flags.writeable:
                    confusion = np.array(confusion, copy=True)
                confusion.flags.writeable = False
            frozen.append(confusion)
        object.__setattr__(self, "confusions", tuple(frozen))


@dataclass(frozen=True)
class ResetStep:
    """Resolved reset of one or more flat subsystems to |0>, with optional condition.

    `Reset` is an `AppliedOperation`, so it can carry a feedforward `condition`
    just like a gate. The lowered form stores it as ``(clbit_index, value)``
    AND-terms; the per-shot loop skips the reset when the guard fails.
    """

    reset_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None


ResolvedStep = ApplyMatrixStep | ApplyChannelStep | MeasurementStep | ResetStep
