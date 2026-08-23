"""Validate one-time all-six-pair binding for a 2 x 2 atom target."""

from math import sqrt

import pytest

import fatqat as fq
from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator.atom_3level import target as target_module
from fatqat.emulator.atom_3level.qutip_adapter import _Atom3LevelQutipAdapter
from fatqat.emulator.atom_3level.target import _Atom3LevelTarget


def test_target_computes_each_public_two_by_two_pair_exactly_once(
    atom_3level_model, monkeypatch
):
    arrangement = fq.emulator.AtomArrangement.rectangular(2, 2, 2.0)
    calls = []
    original_dist = target_module.dist

    def counting_dist(first, second):
        calls.append((first, second))
        return original_dist(first, second)

    monkeypatch.setattr(target_module, "dist", counting_dist)
    target = _Atom3LevelTarget(atom_3level_model, arrangement)
    interactions = target.interactions
    assert len(interactions) == 6
    assert len(calls) == 6
    assert {(value.first, value.second) for value in interactions} == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    expected_radii = {
        (0, 1): 2.0,
        (0, 2): 2.0,
        (0, 3): 2.0 * sqrt(2.0),
        (1, 2): 2.0 * sqrt(2.0),
        (1, 3): 2.0,
        (2, 3): 2.0,
    }
    for value in interactions:
        radius = expected_radii[(value.first, value.second)]
        assert value.distance_um == pytest.approx(radius)
        assert value.signed_strength_rad_per_us * radius**6 == pytest.approx(
            atom_3level_model.c6_angular_per_us_um6, rel=1e-13
        )

    adapter = _Atom3LevelQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(target.device_labels, (3, 3, 3, 3)),
    )
    adapter.interaction_drift()
    adapter.interaction_drift()
    assert len(calls) == 6
    for owner in (target, adapter):
        for removed in (
            "_binding_snapshot",
            "_occupancy",
            "_logical_to_site",
            "_interaction_provider",
            "_interaction_cache",
        ):
            assert not hasattr(owner, removed)
