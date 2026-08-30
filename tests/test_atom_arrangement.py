"""Public regular neutral-atom arrangement value tests."""

from math import inf, nan
from operator import setitem

import pytest

from fatqat.emulator import AtomArrangement


def test_rectangular_arrangement_is_row_major_3d_immutable_value():
    arrangement = AtomArrangement.rectangular(rows=2, cols=3, spacing=1.5)

    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 1.5, 0.0),
        (1.5, 1.5, 0.0),
        (3.0, 1.5, 0.0),
    )
    assert (
        arrangement.num_sites == len(arrangement) == len(arrangement.coordinates) == 6
    )
    assert arrangement.distance_unit == "um"
    assert not hasattr(arrangement, "cardinality")
    assert not hasattr(arrangement, "occupancy")
    assert arrangement == AtomArrangement.rectangular(2, 3, 1.5)
    assert hash(arrangement) == hash(AtomArrangement.rectangular(2, 3, 1.5))

    with pytest.raises((AttributeError, TypeError)):
        setitem(arrangement.coordinates, 0, (99.0, 0.0, 0.0))
    with pytest.raises((AttributeError, TypeError)):
        arrangement.rows = 99


def test_chain_arrangement_is_the_one_row_convenience_geometry():
    arrangement = AtomArrangement.chain(num_sites=3, spacing=1.5)

    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        AtomArrangement.chain(num_sites=0, spacing=1.5)


@pytest.mark.parametrize(
    "rows, cols", [(True, 1), (1, False), (1.0, 1), (1, 2.5), (0, 1), (1, -1)]
)
def test_rectangular_arrangement_rejects_invalid_dimensions(rows, cols):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(rows, cols, 1.0)


@pytest.mark.parametrize("spacing", [True, 0, -1.0, inf, -inf, nan])
def test_rectangular_arrangement_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(1, 1, spacing)
