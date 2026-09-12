"""Public neutral-atom arrangement value tests."""

from math import inf, nan
from operator import setitem

import numpy as np
import pytest

from fatqat.emulator import AtomArrangement


@pytest.mark.parametrize(
    "spacing, expected_spacing, row_spacing",
    [(1.5, 1.5, 1.5), ((1.5, 2), (1.5, 2.0), 2.0), ((1.5, 1.5), (1.5, 1.5), 1.5)],
)
def test_rectangular_arrangement_is_row_major_3d_immutable_value(
    spacing, expected_spacing, row_spacing
):
    arrangement = AtomArrangement.rectangular(rows=2, cols=3, spacing=spacing)

    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, row_spacing, 0.0),
        (1.5, row_spacing, 0.0),
        (3.0, row_spacing, 0.0),
    )
    assert (
        arrangement.num_sites == len(arrangement) == len(arrangement.coordinates) == 6
    )
    assert arrangement.distance_unit == "um"
    assert (arrangement.rows, arrangement.cols, arrangement.spacing) == (
        2,
        3,
        expected_spacing,
    )
    assert arrangement.spatial_dimension == 2
    assert not hasattr(arrangement, "cardinality")
    assert not hasattr(arrangement, "occupancy")
    assert arrangement == AtomArrangement.rectangular(2, 3, spacing)
    assert hash(arrangement) == hash(AtomArrangement.rectangular(2, 3, spacing))

    with pytest.raises((AttributeError, TypeError)):
        setitem(arrangement.coordinates, 0, (99.0, 0.0, 0.0))
    with pytest.raises((AttributeError, TypeError)):
        arrangement.rows = 99


def test_chain_arrangement_is_the_one_row_convenience_geometry():
    arrangement = AtomArrangement.chain(num_sites=3, spacing=1.5)

    assert arrangement == AtomArrangement.rectangular(1, 3, 1.5)
    assert arrangement.spatial_dimension == 2
    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        AtomArrangement.chain(num_sites=0, spacing=1.5)
    with pytest.raises(ValueError):
        AtomArrangement.chain(num_sites=3, spacing=(1.5, 2.0))


@pytest.mark.parametrize(
    "rows, cols", [(True, 1), (1, False), (1.0, 1), (1, 2.5), (0, 1), (1, -1)]
)
def test_rectangular_arrangement_rejects_invalid_dimensions(rows, cols):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(rows, cols, 1.0)


@pytest.mark.parametrize(
    "spacing",
    [
        True,
        0,
        -1.0,
        inf,
        -inf,
        nan,
        (),
        (1,),
        (1, 2, 3),
        (True, 1),
        (1, 0),
        (nan, 1),
        (1, inf),
        "12",
        {1, 2},
        {"x": 1, "y": 2},
    ],
)
def test_rectangular_arrangement_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(1, 1, spacing)


@pytest.mark.parametrize("container", [list, iter, np.asarray])
def test_rectangular_spacing_pair_accepts_ordered_iterables(container):
    arrangement = AtomArrangement.rectangular(2, 2, container([2, 3]))

    assert arrangement.spacing == (2.0, 3.0)
    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (2.0, 3.0, 0.0),
    )


@pytest.mark.parametrize(
    "coordinates, dimension, expected",
    [
        ([(3, -5), (0, 0)], 2, ((3.0, -5.0, 0.0), (0.0, 0.0, 0.0))),
        ([(3, -5, 0), (0, 0, 0)], 3, ((3.0, -5.0, 0.0), (0.0, 0.0, 0.0))),
        ([(3, -5, 4), (0, 0, -2)], 3, ((3.0, -5.0, 4.0), (0.0, 0.0, -2.0))),
    ],
)
def test_explicit_coordinates_preserve_order_and_input_dimension(
    coordinates, dimension, expected
):
    arrangement = AtomArrangement.from_coordinates(coordinates)

    assert arrangement.coordinates == expected
    assert arrangement.spatial_dimension == dimension
    assert arrangement.num_sites == len(arrangement) == len(coordinates)
    assert arrangement.distance_unit == "um"
    assert (arrangement.rows, arrangement.cols, arrangement.spacing) == (
        None,
        None,
        None,
    )
    assert arrangement == AtomArrangement.from_coordinates(coordinates)
    assert hash(arrangement) == hash(AtomArrangement.from_coordinates(coordinates))


@pytest.mark.parametrize("container", [list, tuple, iter, np.asarray])
def test_explicit_coordinates_accept_ordered_iterables(container):
    arrangement = AtomArrangement.from_coordinates(container([(6, 0), (0, 0)]))

    assert arrangement.coordinates == ((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert all(
        isinstance(value, float) for point in arrangement.coordinates for value in point
    )


def test_explicit_coordinates_copy_input_and_remain_immutable():
    coordinates = [[0, 0, 6]]
    arrangement = AtomArrangement.from_coordinates(coordinates)
    coordinates[0][2] = 9
    coordinates.append([0, 0, 12])

    assert arrangement.coordinates == ((0.0, 0.0, 6.0),)
    assert arrangement.num_sites == 1
    with pytest.raises((AttributeError, TypeError)):
        setitem(arrangement.coordinates, 0, (0.0, 0.0, 9.0))
    with pytest.raises((AttributeError, TypeError)):
        arrangement.spatial_dimension = 2


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ([], "at least one site"),
        (None, "ordered iterable"),
        ("12", "ordered iterable"),
        ({(0, 0), (1, 1)}, "ordered iterable"),
        ({(0, 0): "site"}, "ordered iterable"),
        ([None], "ordered iterable"),
        (["12"], "ordered iterable"),
        ([{0, 1}], "ordered iterable"),
        ([{0: 1, 2: 3}], "ordered iterable"),
        ([()], "two or three components"),
        ([(0,)], "two or three components"),
        ([(0, 1, 2, 3)], "two or three components"),
        ([(0, 0), (1, 1, 1)], "must have 2 components"),
        ([(0, 0, 0), (1, 1)], "must have 3 components"),
        ([(0, 0), (0.0, -0.0)], "distinct positions"),
        ([(0, 0, 0), (0, 0, 0)], "distinct positions"),
        ([(2**54, 0), (2**54 + 1, 0)], "distinct positions"),
    ],
)
def test_explicit_coordinates_reject_invalid_layouts(coordinates, message):
    with pytest.raises(ValueError, match=message):
        AtomArrangement.from_coordinates(coordinates)


@pytest.mark.parametrize(
    "component", [True, np.bool_(False), None, "1", 1j, inf, -inf, nan, 10**400]
)
def test_explicit_coordinates_reject_invalid_components(component):
    with pytest.raises(ValueError, match=r"coordinates\[0\]\[2\].*finite real number"):
        AtomArrangement.from_coordinates([(0, 0, component)])


@pytest.mark.parametrize("dimension", [2, 3])
def test_combining_arrangements_concatenates_sites_and_preserves_dimension(dimension):
    first = AtomArrangement.from_coordinates([(4, 2), (3, 4)])
    second = AtomArrangement.chain(2, spacing=1.0)
    third = AtomArrangement.from_coordinates([(9, 9, 0)[:dimension]])

    combined = AtomArrangement.combine(first, second, third)

    assert combined.coordinates == (
        (4.0, 2.0, 0.0),
        (3.0, 4.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (9.0, 9.0, 0.0),
    )
    assert combined.num_sites == len(combined) == 5
    assert combined.spatial_dimension == dimension
    assert (combined.rows, combined.cols, combined.spacing) == (None, None, None)
    assert combined.distance_unit == "um"
    assert combined.groups == {}


def test_combining_arrangements_rejects_overlapping_sites():
    first = AtomArrangement.chain(2, spacing=1.0)
    second = AtomArrangement.from_coordinates([(3, 0, 0), (1, 0, 0)])

    with pytest.raises(ValueError):
        AtomArrangement.combine(first, second)


def test_coordinate_groups_use_exact_xy_and_final_site_order_ignoring_z():
    nearby_x = np.nextafter(2.0, 3.0)
    nearby_y = np.nextafter(1.0, 2.0)
    first = AtomArrangement.from_coordinates([(2, 3, 1), (1, 1, 0)])
    second = AtomArrangement.from_coordinates(
        [(2, 3, 7), (1, 3, 0), (nearby_x, nearby_y, 0)]
    )

    combined = AtomArrangement.combine(first, second)

    assert tuple(combined.row_groups.items()) == (
        (3.0, (0, 2, 3)),
        (1.0, (1,)),
        (nearby_y, (4,)),
    )
    assert tuple(combined.column_groups.items()) == (
        (2.0, (0, 2)),
        (1.0, (1, 3)),
        (nearby_x, (4,)),
    )
    for groups in (combined.row_groups, combined.column_groups):
        with pytest.raises(TypeError):
            setitem(groups, 0.0, (99,))
        with pytest.raises(TypeError):
            setitem(next(iter(groups.values())), 0, 99)


@pytest.mark.parametrize("count", [0, 1])
def test_combining_arrangements_requires_at_least_two_inputs(count):
    arrangements = (AtomArrangement.from_coordinates([(0, 0)]),) * count

    with pytest.raises(ValueError):
        AtomArrangement.combine(*arrangements)


def test_combining_arrangements_rejects_raw_coordinate_inputs():
    first = AtomArrangement.from_coordinates([(0, 0)])

    with pytest.raises(TypeError):
        AtomArrangement.combine(first, [(2, 0)])


@pytest.mark.parametrize(
    "factory",
    [
        lambda label: AtomArrangement.chain(2, 1.0, label=label),
        lambda label: AtomArrangement.rectangular(1, 2, 1.0, label=label),
        lambda label: AtomArrangement.from_coordinates([(0, 0), (1, 0)], label=label),
        lambda label: AtomArrangement.combine(
            AtomArrangement.from_coordinates([(0, 0)]),
            AtomArrangement.from_coordinates([(1, 0)]),
            label=label,
        ),
    ],
    ids=("chain", "rectangular", "coordinates", "combine"),
)
def test_factory_labels_name_all_sites_and_preserve_label_text(factory):
    arrangement = factory(" left ")

    assert arrangement.label == " left "
    assert arrangement.groups == {" left ": (0, 1)}
    assert arrangement.group(" left ") == (0, 1)
    with pytest.raises(KeyError):
        arrangement.group("left")
    with pytest.raises(ValueError):
        factory(" \n ")


@pytest.mark.parametrize("label", ["", 1])
def test_arrangement_labels_reject_empty_or_non_string_values(label):
    with pytest.raises(ValueError):
        AtomArrangement.from_coordinates([(0, 0)], label=label)


def test_nested_combination_preserves_reindexed_immutable_groups():
    left = AtomArrangement.chain(2, spacing=2.0, label="left")
    right = AtomArrangement.from_coordinates([(6, 0), (8, 0)], label="right")
    pair = AtomArrangement.combine(left, right, label="pair")
    prefix = AtomArrangement.from_coordinates([(-2, 0)])
    suffix = AtomArrangement.from_coordinates([(10, 0)], label="suffix")

    combined = AtomArrangement.combine(prefix, pair, suffix, label="whole")

    assert combined.groups == {
        "left": (1, 2),
        "right": (3, 4),
        "pair": (1, 2, 3, 4),
        "suffix": (5,),
        "whole": (0, 1, 2, 3, 4, 5),
    }
    assert combined.group("pair") == (1, 2, 3, 4)
    assert pair.groups == {"left": (0, 1), "right": (2, 3), "pair": (0, 1, 2, 3)}
    with pytest.raises(TypeError):
        setitem(combined.groups, "left", (99,))
    with pytest.raises(TypeError):
        setitem(combined.group("left"), 0, 99)
    with pytest.raises(AttributeError):
        combined.label = "changed"


@pytest.mark.parametrize("collision", ["child", "composite", "nested"])
def test_combining_arrangements_rejects_duplicate_group_labels(collision):
    first = AtomArrangement.from_coordinates([(0, 0)], label="left")
    second = AtomArrangement.from_coordinates(
        [(1, 0)], label="right" if collision == "composite" else "left"
    )
    if collision == "nested":
        first = AtomArrangement.combine(
            first, AtomArrangement.from_coordinates([(2, 0)])
        )

    with pytest.raises(ValueError):
        AtomArrangement.combine(
            first, second, label="left" if collision == "composite" else None
        )
