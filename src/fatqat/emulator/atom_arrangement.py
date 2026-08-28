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
    """An immutable regular arrangement of physical atom sites.

    Construct arrangements with ``chain()`` or ``rectangular()``.
    Coordinates are in micrometres and ordered row-major, with the x
    coordinate advancing across columns. Arbitrary-coordinate construction is
    not part of the current public API.

    Attributes:
        rows: Positive number of rows.
        cols: Positive number of columns.
        spacing: Uniform horizontal/vertical nearest-neighbor spacing in
            micrometres for the current atom model families.
        coordinates: Immutable row-major ``(x, y, z)`` coordinates with
            ``z == 0.0``.

    Examples:
        >>> import fatqat as fq
        >>> arrangement = fq.emulator.AtomArrangement.chain(3, spacing=6.0)
        >>> arrangement.num_sites
        3
        >>> arrangement.coordinates[2]
        (12.0, 0.0, 0.0)
    """

    rows: int
    cols: int
    spacing: float
    coordinates: tuple[tuple[float, float, float], ...]

    @classmethod
    def chain(cls, num_sites: int, spacing: float) -> AtomArrangement:
        """Create a one-dimensional chain ordered along the x axis.

        Args:
            num_sites: Positive number of physical sites.
            spacing: Positive finite nearest-neighbor spacing in micrometres.

        Returns:
            An immutable one-row arrangement with ``num_sites`` coordinates.

        Raises:
            ValueError: If ``num_sites`` is not a positive integer or spacing
                is not a positive finite real number. Booleans are rejected.
        """
        num_sites = _dimension(num_sites, "num_sites")
        return cls.rectangular(rows=1, cols=num_sites, spacing=spacing)

    @classmethod
    def rectangular(cls, rows: int, cols: int, spacing: float) -> AtomArrangement:
        """Create a row-major rectangular arrangement with zero z coordinates.

        Args:
            rows: Positive number of rows.
            cols: Positive number of columns.
            spacing: Positive finite nearest-neighbor spacing in micrometres
                for the current neutral-atom model families.

        Returns:
            An immutable arrangement with ``rows * cols`` declared sites and
            coordinates ``(column * spacing, row * spacing, 0)``.

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
    def num_sites(self) -> int:
        """Return the exact number of declared physical sites.

        Returns:
            The number of coordinates in this immutable geometry. This is a
            site count, not a dynamic atom-occupancy count.
        """
        return len(self.coordinates)

    def __len__(self) -> int:
        """Return the number of sites.

        Returns:
            The same site count as ``num_sites``.
        """
        return self.num_sites
