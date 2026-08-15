"""Rectangular interaction-policy tests."""

import pytest

from fatqat import AtomArrangement
from fatqat.emulator.atom_2level import GridInteractionPolicy
from fatqat.emulator.atom_2level import policy as policy_module


def test_public_constructor_is_disabled_and_factories_are_exact():
    with pytest.raises(TypeError):
        GridInteractionPolicy("nearest_neighbor")

    assert GridInteractionPolicy.nearest_neighbor().mode == "nearest_neighbor"
    assert GridInteractionPolicy.full_pair().mode == "full_pair"


def test_nearest_neighbor_edges_are_unique_canonical_and_row_major():
    arrangement = AtomArrangement.rectangular(2, 3, 2)

    assert policy_module._interaction_edges(
        GridInteractionPolicy.nearest_neighbor(), arrangement
    ) == ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5))


def test_full_pair_edges_include_every_i_less_than_j():
    arrangement = AtomArrangement.rectangular(2, 2, 2)

    assert policy_module._interaction_edges(
        GridInteractionPolicy.full_pair(), arrangement
    ) == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def test_nearest_neighbor_never_calls_full_pair_builder(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("nearest-neighbor construction scanned all pairs")

    monkeypatch.setattr(policy_module, "_full_pair_edges", forbidden)
    arrangement = AtomArrangement.rectangular(50, 60, 1)
    edges = policy_module._interaction_edges(
        GridInteractionPolicy.nearest_neighbor(), arrangement
    )

    assert len(edges) == 50 * 59 + 49 * 60
