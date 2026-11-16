"""Classical measurement reporting for the shared Python workflow.

Physical collapse stays with the numerical component. These helpers map physical
outcomes to reported digits, apply readout confusion, and retain last-writer
semantics for deferred measurements.
"""

from __future__ import annotations

import numpy as np

from ...._backends.steps import MeasurementStep, ResolvedStep

ERASURE_DIGIT = 2


def _map_physical_digit(physical_digit: int, reported_digit_map) -> int:
    """Map one physical measurement outcome to its reported classical digit."""
    if reported_digit_map is None:
        return physical_digit
    return reported_digit_map[physical_digit]


def _report_digit(reported_digit: int, confusion, rng: np.random.Generator) -> int:
    """Optionally resample one reported digit through readout confusion.

    ``confusion`` is column-stochastic (``C[i, j] = P(report i | mapped j)``)
    or ``None`` for an error-free readout. Only the reported classical value
    is affected; the caller's collapsed state keeps the physical outcome.
    """
    if confusion is None:
        return reported_digit
    return int(rng.choice(confusion.shape[0], p=confusion[:, reported_digit]))


def _reporting_by_clbit(
    plan: list[ResolvedStep],
) -> dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]]:
    """Map each clbit to the reporting contract of its last writer.

    Fast-path counts decode only each clbit's final value (later measurement
    writes replace earlier ones, see ``_decode_engine_indices_to_clbit_rows``),
    so the
    physical-to-reported map and its readout confusion must likewise come from
    the last writer. An earlier map or confusion must not survive an error-free
    overwrite.
    """
    by_clbit: dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]] = {}
    for step in plan:
        if isinstance(step, MeasurementStep):
            confusions = step.confusions or (None,) * len(step.classical_indices)
            maps = step.reported_digit_maps or (None,) * len(step.classical_indices)
            for clbit, reported_map, confusion in zip(
                step.classical_indices, maps, confusions
            ):
                by_clbit[clbit] = (reported_map, confusion)
    return by_clbit


def _apply_measurement_reporting(
    rows: np.ndarray,
    by_clbit: dict[int, tuple[tuple[int, ...] | None, np.ndarray | None]],
    rng: np.random.Generator,
) -> None:
    """Map physical decoded columns, then apply readout confusion in place."""
    for clbit, (reported_map, confusion) in by_clbit.items():
        if reported_map is not None:
            rows[:, clbit] = np.asarray(reported_map, dtype=int)[rows[:, clbit]]
        if confusion is None:
            continue
        dim = confusion.shape[0]
        reported_column = rows[:, clbit].copy()
        for reported_digit in range(dim):
            mask = reported_column == reported_digit
            hits = int(mask.sum())
            if hits:
                rows[mask, clbit] = rng.choice(
                    dim, size=hits, p=confusion[:, reported_digit]
                )
