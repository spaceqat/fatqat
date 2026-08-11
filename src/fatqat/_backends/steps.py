"""Resolved execution-plan step types shared by every backend and engine.

A :py:class:`~fatqat.Program` lowers to a `list[ResolvedStep]`: each step is either an
`ApplyMatrixStep` (a local matrix payload built from a matrix-implementation
rule plus layout-resolved target indices), an `ApplyChannelStep` (a Kraus
payload built from a channel-implementation rule, for channel-representable
noise), a `MeasurementStep`, or a `ResetStep`. Defined here, separate from
both `implementation/` / `noise/` (the rule protocols that only ever produce
bare arrays, never a step) and the `simulator/` / `emulator/` packages (which
would otherwise need to import this type from each other and cycle), so every
backend and every engine can import it without depending on one another.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class BuiltinKernelKey(enum.Enum):
    """Canonical identity of a built-in gate implementation, one member per gate.

    Carried on `ApplyMatrixStep` so an engine can select a specialized kernel
    by declared identity instead of inspecting the matrix. Keys are attached
    only at canonical registration in ``default_matrix_implementation_map()``
    - never inferred from the operation class or by comparing matrices - so a
    custom implementation is always ``None``-keyed and takes an engine's
    content-inspecting fallback path.

    Deliberately one member per gate family rather than one per kernel:
    which gates *share* a kernel is an engine-side, per-representation fact
    that may be re-partitioned at any time (a future density-matrix engine
    will group differently than the statevector one), while identity is
    permanent. Sharing lives in the engine's key-to-kernel table, never here.
    """

    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    H = enum.auto()
    I = enum.auto()
    S = enum.auto()
    SDG = enum.auto()
    SX = enum.auto()
    T = enum.auto()
    TDG = enum.auto()
    CX = enum.auto()
    CZ = enum.auto()
    SWAP = enum.auto()
    CY = enum.auto()
    CS = enum.auto()
    ISWAP = enum.auto()
    CCX = enum.auto()
    CSWAP = enum.auto()
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    PHASE = enum.auto()
    CPHASE = enum.auto()
    SHIFT = enum.auto()
    CLOCK = enum.auto()
    SUM = enum.auto()
    SWAP_LEVELS = enum.auto()
    FOURIER = enum.auto()
    FOURIERDG = enum.auto()
    SUBSPACE_RX = enum.auto()
    SUBSPACE_RY = enum.auto()
    SUBSPACE_RZ = enum.auto()
    CCLOCK = enum.auto()


@dataclass(frozen=True)
class ApplyMatrixStep:
    """Resolved local matrix payload consumed by the statevector engine.

    Doubles as the "apply a matrix" entry in a backend execution plan and as the
    payload the engine applies. The matrix is marked read-only after construction
    so this frozen value object cannot be mutated through the NumPy array buffer.

    Attributes:
        matrix: Local operation matrix. Always the numeric source of truth:
            a specialized kernel selected via ``kernel_key`` still reads its
            numbers (a rotation's phases, a permutation's entries) from here.
        target_indices: Flat subsystem indices the matrix acts on.
        condition: Optional feedforward guard as lowered ``(clbit_index, value)``
            AND-terms. ``None`` means unconditional. The engine ignores this
            field; the backend's per-shot loop evaluates it.
        kernel_key: Canonical identity of the built-in implementation that
            produced ``matrix``, or ``None`` for custom/device
            implementations. Engines may use it to select a specialized
            kernel at plan-preparation time; ignoring it is always correct.
    """

    matrix: np.ndarray
    target_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None
    kernel_key: BuiltinKernelKey | None = None

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
class AtomLossStep:
    """Per-atom classical loss: roll one die per target, partial-trace on hit.

    Emitted right after the parent gate's `ApplyMatrixStep`, same slot as
    `ApplyChannelStep`. Carries no arrays — loss is classical.

    Attributes:
        target_indices: Flat subsystem indices, one independent die each.
        p: Per-atom loss probability in ``[0, 1]``.
        condition: The parent gate's lowered feedforward guard, or ``None``.
    """

    target_indices: tuple[int, ...]
    p: float
    condition: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class OccupancyInitStep:
    """Front-of-plan marker carrying the shot's initial occupied subsystems.

    Atom-grid lowering emits this first so the per-shot loop can seed its
    occupancy set from the loaded atoms. It touches no state and emits no
    physics; engines that ignore it are unaffected.

    Attributes:
        occupied_indices: Flat subsystem indices occupied at shot start.
    """

    occupied_indices: tuple[int, ...]


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat subsystem indices into matching flat clbit indices.

    A measurement has three deliberately separate stages: physical
    measurement/collapse, physical-outcome-to-reported-digit mapping, then
    optional classical readout confusion. ``reported_digit_maps`` contains
    one map per measured subsystem, where a map's index is a physical outcome
    and its value is the reported classical digit. ``None`` retains the
    identity-map compatibility default; normal matrix lowering supplies the
    identity maps explicitly.

    ``confusions`` carries classical readout error resolved from the noise
    model at lowering: one optional column-stochastic confusion matrix per
    measured subsystem (aligned with ``measured_indices``), or ``None`` when
    no readout error applies. Its dimensions are those of the reported
    classical digit, not necessarily the engine's physical subsystem. The
    physical collapse always uses the physical outcome; only the value written
    to the classical register is mapped and resampled, so state export and
    subsystem reuse are untouched and execution-strategy classification never
    changes.
    """

    measured_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]
    confusions: tuple[np.ndarray | None, ...] | None = None
    reported_digit_maps: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.measured_indices) != len(self.classical_indices):
            raise ValueError("measurement subsystem and classical-index counts differ")
        if self.reported_digit_maps is not None:
            if len(self.reported_digit_maps) != len(self.measured_indices):
                raise ValueError(
                    "measurement reported-digit-map count must match measured subsystems"
                )
            frozen_maps: list[tuple[int, ...]] = []
            for reported_map in self.reported_digit_maps:
                reported_map = tuple(reported_map)
                if not reported_map:
                    raise ValueError(
                        "measurement reported-digit maps must not be empty"
                    )
                if any(
                    not isinstance(digit, (int, np.integer)) or isinstance(digit, bool)
                    for digit in reported_map
                ) or any(digit < 0 for digit in reported_map):
                    raise ValueError(
                        "measurement reported-digit maps must contain non-negative integers"
                    )
                frozen_maps.append(tuple(int(digit) for digit in reported_map))
            object.__setattr__(self, "reported_digit_maps", tuple(frozen_maps))

        # Same freezing policy as the array-carrying steps above.
        if self.confusions is None:
            return
        if len(self.confusions) != len(self.measured_indices):
            raise ValueError(
                "measurement confusion count must match measured subsystems"
            )
        frozen: list[np.ndarray | None] = []
        maps = self.reported_digit_maps or (None,) * len(self.measured_indices)
        for confusion, reported_map in zip(self.confusions, maps):
            if confusion is not None:
                if reported_map is not None:
                    reported_dim = max(reported_map) + 1
                    if confusion.shape != (reported_dim, reported_dim):
                        raise ValueError(
                            "measurement confusion shape must match the reported "
                            f"classical dimension {reported_dim}, got {confusion.shape}"
                        )
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


ResolvedStep = ApplyMatrixStep | ApplyChannelStep | AtomLossStep | OccupancyInitStep | MeasurementStep | ResetStep
