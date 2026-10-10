"""Immutable public arrangements for neutral-atom emulators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, ClassVar


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a positive finite real number, not bool"
        ) from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite real number, not bool")
    return result


def _ordered_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be an ordered iterable")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered iterable") from exc


def _spacing(value: Any) -> float | tuple[float, float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return _positive_number(value, "spacing")
    components = _ordered_tuple(value, "spacing")
    if len(components) != 2:
        raise ValueError("spacing must contain exactly two components")
    return (
        _positive_number(components[0], "spacing[0]"),
        _positive_number(components[1], "spacing[1]"),
    )


def _label(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("label must be None or a non-empty string")
    return value


def _coordinate(value: Any, name: str) -> tuple[float, ...]:
    components = _ordered_tuple(value, name)
    if len(components) not in (2, 3):
        raise ValueError(f"{name} must contain exactly two or three components")
    converted = []
    for index, component in enumerate(components):
        error = f"{name}[{index}] must be a finite real number, not bool"
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ValueError(error)
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error) from exc
        if not isfinite(number):
            raise ValueError(error)
        converted.append(number)
    return tuple(converted)


@dataclass(frozen=True, init=False)
class AtomArrangement:
    """An immutable arrangement of physical atom sites in micrometres.

    Use ``from_coordinates()`` for explicit two- or three-dimensional sites,
    or ``chain()`` and ``rectangular()`` for regular layouts. Coordinates are
    always stored as immutable ``(x, y, z)`` tuples. Explicit coordinates keep
    their input order; regular layouts use row-major order with x advancing
    across columns. Coordinate order defines the site indices.

    Attributes:
        rows: Positive number of rows for a regular layout, otherwise None.
        cols: Positive number of columns for a regular layout, otherwise None.
        spacing: Uniform horizontal/vertical nearest-neighbor spacing in
            micrometres for a regular layout, or an ``(x, y)`` spacing pair.
            Explicit-coordinate and combined layouts use None.
        coordinates: Immutable ``(x, y, z)`` coordinates in micrometres.
            Two-dimensional input has ``z == 0.0``.
        spatial_dimension: Coordinate-space dimension, 2 or 3, determined by
            the number of input components. Explicit z values, including
            zero, give 3. Regular layouts use 2, including chains embedded
            in the xy plane. This is independent of the atom's energy levels.
        label: Optional non-empty label for this complete layout. When layouts
            are combined, labels identify immutable site-index groups.
        row_groups: Read-only mapping from each y coordinate to site indices.
        column_groups: Read-only mapping from each x coordinate to site indices.

    Examples:
        >>> import fatqat as fq
        >>> arrangement = fq.emulator.AtomArrangement.chain(3, spacing=6.0)
        >>> arrangement.num_sites
        3
        >>> arrangement.coordinates[2]
        (12.0, 0.0, 0.0)
    """

    rows: int | None
    cols: int | None
    spacing: float | tuple[float, float] | None
    coordinates: tuple[tuple[float, float, float], ...]
    spatial_dimension: int
    label: str | None
    _groups: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)
    distance_unit: ClassVar[str] = "um"

    @classmethod
    def from_coordinates(
        cls, coordinates: Iterable[Iterable[Real]], *, label: str | None = None
    ) -> AtomArrangement:
        """Create a fixed layout from ordered two- or three-dimensional sites.

        Args:
            coordinates: Nonempty ordered iterable of coordinate iterables
                in micrometres. Every site must have exactly two components
                ``(x, y)`` or every site exactly three ``(x, y, z)``. Components
                must be finite real numbers, excluding booleans; negative
                values are allowed. Positions must be distinct after float
                conversion. Strings, mappings, and sets are not accepted as
                the outer iterable or as individual coordinates.
            label: Optional non-empty label. Whitespace-only labels are
                rejected. The original string is preserved.

        Returns:
            AtomArrangement: An immutable copy of the coordinates in input
                order, with zero z values added to two-dimensional input.
                ``spatial_dimension`` records the input dimension, including
                3 when all explicit z values are zero. ``rows``, ``cols``, and
                ``spacing`` are None, even if the sites form a regular grid.

        Raises:
            ValueError: If the input is empty, unordered, has invalid or mixed
                dimensions, contains invalid components, repeats a position,
                or the label is invalid.

        Examples:
            >>> import fatqat as fq
            >>> sites = fq.emulator.AtomArrangement.from_coordinates(
            ...     [(3, 5), (0, 0)]
            ... )
            >>> sites.coordinates
            ((3.0, 5.0, 0.0), (0.0, 0.0, 0.0))
            >>> sites.spatial_dimension
            2
        """
        label = _label(label)
        points = _ordered_tuple(coordinates, "coordinates")
        if not points:
            raise ValueError("coordinates must contain at least one site")
        normalized = []
        spatial_dimension = None
        for index, point in enumerate(points):
            values = _coordinate(point, f"coordinates[{index}]")
            if spatial_dimension is None:
                spatial_dimension = len(values)
            elif len(values) != spatial_dimension:
                raise ValueError(
                    f"coordinates[{index}] must have {spatial_dimension} components "
                    "like every other site"
                )
            normalized.append(
                (values[0], values[1], values[2] if len(values) == 3 else 0.0)
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("coordinates must contain distinct positions")

        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", None)
        object.__setattr__(instance, "cols", None)
        object.__setattr__(instance, "spacing", None)
        object.__setattr__(instance, "coordinates", tuple(normalized))
        object.__setattr__(instance, "spatial_dimension", spatial_dimension)
        object.__setattr__(instance, "label", label)
        groups = ((label, tuple(range(len(normalized)))),) if label is not None else ()
        object.__setattr__(instance, "_groups", groups)
        return instance

    @classmethod
    def chain(
        cls, num_sites: int, spacing: Real, *, label: str | None = None
    ) -> AtomArrangement:
        """Create a one-dimensional chain ordered along the x axis.

        Args:
            num_sites: Positive number of physical sites.
            spacing: Positive finite nearest-neighbor spacing in micrometres.
            label: Optional non-empty label for the complete chain.

        Returns:
            An immutable one-row arrangement with ``num_sites`` coordinates.

        Raises:
            ValueError: If ``num_sites`` is not a positive integer or spacing
                is not a positive finite real number, or the label is invalid.
                Booleans are rejected.
        """
        num_sites = _dimension(num_sites, "num_sites")
        spacing = _positive_number(spacing, "spacing")
        return cls.rectangular(rows=1, cols=num_sites, spacing=spacing, label=label)

    @classmethod
    def rectangular(
        cls,
        rows: int,
        cols: int,
        spacing: Real | Iterable[Real],
        *,
        label: str | None = None,
    ) -> AtomArrangement:
        """Create a row-major rectangular arrangement with zero z coordinates.

        Args:
            rows: Positive number of rows.
            cols: Positive number of columns.
            spacing: Positive finite nearest-neighbor spacing in micrometres,
                or an ordered ``(x_spacing, y_spacing)`` pair of positive
                finite real numbers. Booleans are rejected.
            label: Optional non-empty label for the complete rectangle.

        Returns:
            An immutable arrangement with ``rows * cols`` declared sites and
            coordinates ``(column * x_spacing, row * y_spacing, 0)``.

        Raises:
            ValueError: If a dimension is not a positive integer, spacing is
                not a valid positive scalar or pair, or the label is invalid.
                Booleans are rejected.
        """
        rows = _dimension(rows, "rows")
        cols = _dimension(cols, "cols")
        spacing = _spacing(spacing)
        x_spacing, y_spacing = (
            (spacing, spacing) if isinstance(spacing, float) else spacing
        )
        coordinates = tuple(
            (column * x_spacing, row * y_spacing, 0.0)
            for row in range(rows)
            for column in range(cols)
        )
        label = _label(label)
        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", rows)
        object.__setattr__(instance, "cols", cols)
        object.__setattr__(instance, "spacing", spacing)
        object.__setattr__(instance, "coordinates", coordinates)
        object.__setattr__(instance, "spatial_dimension", 2)
        object.__setattr__(instance, "label", label)
        groups = ((label, tuple(range(len(coordinates)))),) if label is not None else ()
        object.__setattr__(instance, "_groups", groups)
        return instance

    @classmethod
    def combine(
        cls, *arrangements: AtomArrangement, label: str | None = None
    ) -> AtomArrangement:
        """Combine two or more layouts in argument and site order.

        Coordinates are concatenated rather than treated as an unordered set.
        Child groups retain their labels and receive indices in the combined
        layout. A label supplied to this method adds a group spanning the
        complete result, so nested combinations can retain their hierarchy.

        Args:
            *arrangements: At least two AtomArrangement values in the desired
                site order. Their coordinates must not overlap.
            label: Optional non-empty label for the complete combined layout.
                It must be distinct from every retained child-group label.

        Returns:
            AtomArrangement: A combined layout whose ``rows``, ``cols``, and
                ``spacing`` are None. Its spatial dimension is 3 if any input
                is 3D and otherwise 2.

        Raises:
            TypeError: If an input is not an AtomArrangement.
            ValueError: If fewer than two arrangements are supplied, positions
                overlap, the label is invalid, or group labels are duplicated.
        """
        if len(arrangements) < 2:
            raise ValueError("combine requires at least two arrangements")
        label = _label(label)
        coordinates = []
        groups = []
        group_names = set()
        spatial_dimension = 2
        for index, arrangement in enumerate(arrangements):
            if not isinstance(arrangement, cls):
                raise TypeError(f"arrangements[{index}] must be an AtomArrangement")
            offset = len(coordinates)
            coordinates.extend(arrangement.coordinates)
            spatial_dimension = max(spatial_dimension, arrangement.spatial_dimension)
            for group_name, site_indices in arrangement._groups:
                if group_name in group_names:
                    raise ValueError(
                        f"duplicate arrangement group label {group_name!r}"
                    )
                group_names.add(group_name)
                groups.append(
                    (group_name, tuple(offset + site for site in site_indices))
                )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("combined arrangements must contain distinct positions")
        if label is not None:
            if label in group_names:
                raise ValueError(f"duplicate arrangement group label {label!r}")
            groups.append((label, tuple(range(len(coordinates)))))

        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", None)
        object.__setattr__(instance, "cols", None)
        object.__setattr__(instance, "spacing", None)
        object.__setattr__(instance, "coordinates", tuple(coordinates))
        object.__setattr__(instance, "spatial_dimension", spatial_dimension)
        object.__setattr__(instance, "label", label)
        object.__setattr__(instance, "_groups", tuple(groups))
        return instance

    @property
    def groups(self) -> Mapping[str, tuple[int, ...]]:
        """Return the immutable labeled site groups in insertion order."""
        return MappingProxyType(dict(self._groups))

    def group(self, label: str) -> tuple[int, ...]:
        """Return the site indices associated with a label.

        Args:
            label: Exact label to look up.

        Returns:
            Immutable site indices in arrangement order.

        Raises:
            KeyError: If the label is unknown.
        """
        try:
            return dict(self._groups)[label]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown arrangement group {label!r}") from exc

    @property
    def row_groups(self) -> Mapping[float, tuple[int, ...]]:
        """Group site indices by exact y coordinate in first-seen order.

        Returns:
            Read-only y-to-indices mapping. Group membership uses the xy
            projection, so z coordinates do not affect it.
        """
        return self._axis_groups(1)

    @property
    def column_groups(self) -> Mapping[float, tuple[int, ...]]:
        """Group site indices by exact x coordinate in first-seen order.

        Returns:
            Read-only x-to-indices mapping. Group membership uses the xy
            projection, so z coordinates do not affect it.
        """
        return self._axis_groups(0)

    def _axis_groups(self, axis: int) -> Mapping[float, tuple[int, ...]]:
        groups: dict[float, list[int]] = {}
        for site, coordinate in enumerate(self.coordinates):
            groups.setdefault(coordinate[axis], []).append(site)
        return MappingProxyType(
            {coordinate: tuple(sites) for coordinate, sites in groups.items()}
        )

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
