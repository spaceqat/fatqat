"""Immutable public arrangements for neutral-atom emulators."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _spacing(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("spacing must be a positive finite number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError("spacing must be a positive finite number")
    return result


@dataclass(frozen=True, init=False)
class AtomArrangement:
    """An immutable rectangular arrangement of initially occupied atom sites.

    Construct arrangements with :meth:`rectangular`. The current three-level
    and two-level atom emulators expose no arbitrary-coordinate public
    constructor. Coordinates are in micrometres and ordered row-major, with
    the x coordinate advancing across columns.

    Attributes:
        rows: Positive number of rows.
        cols: Positive number of columns.
        spacing: Uniform horizontal/vertical nearest-neighbor spacing in
            micrometres for the current atom model families.
        coordinates: Immutable row-major ``(x, y, z)`` coordinates with
            ``z == 0.0``.

    Examples:
        >>> import fatqat as fq
        >>> arrangement = fq.AtomArrangement.rectangular(2, 3, 6.0)
        >>> arrangement.cardinality
        6
        >>> arrangement.coordinates[3]
        (0.0, 6.0, 0.0)
    """

    rows: int
    cols: int
    spacing: float
    coordinates: tuple[tuple[float, float, float], ...]

    @classmethod
    def rectangular(cls, rows: int, cols: int, spacing: float) -> AtomArrangement:
        """Create a row-major rectangular arrangement with zero z coordinates.

        Args:
            rows: Positive number of rows.
            cols: Positive number of columns.
            spacing: Positive finite nearest-neighbor spacing in micrometres
                for the current neutral-atom model families.

        Returns:
            An immutable arrangement with ``rows * cols`` initially occupied
            sites and coordinates ``(column * spacing, row * spacing, 0)``.

        Raises:
            ValueError: If a dimension is not a positive integer or spacing
                is not a positive finite real number. Booleans are rejected.
        """
        rows = _dimension(rows, "rows")
        cols = _dimension(cols, "cols")
        spacing = _spacing(spacing)
        coordinates = tuple(
            (column * spacing, row * spacing, 0.0)
            for row in range(rows)
            for column in range(cols)
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", rows)
        object.__setattr__(instance, "cols", cols)
        object.__setattr__(instance, "spacing", spacing)
        object.__setattr__(instance, "coordinates", coordinates)
        return instance

    @property
    def cardinality(self) -> int:
        """Number of occupied sites, equal to ``rows * cols``.

        This count is public geometry information and does not expose the
        private engine indices assigned during a run.
        """
        return len(self.coordinates)

    def __len__(self) -> int:
        return self.cardinality
