import json
from pathlib import Path

import pytest

from fatqat.compiler.algorithms import (
    compile_interactions as exported_compile_interactions,
)
from fatqat.compiler.algorithms.zap import (
    ZapInteraction,
    compile_interactions,
    load_architecture,
)

GOLDEN_PATH = Path(__file__).with_name("fixtures") / "zap_golden_trace.json"
INTERACTIONS = (
    ZapInteraction("g0", (0,)),
    ZapInteraction("g1", (0, 1)),
    ZapInteraction("g2", (2,)),
    ZapInteraction("g3", (1, 2)),
)


def _small_architecture():
    return {
        "operation_duration": {"1qGate": 1, "2qGate": 1, "atom_transfer": 1},
        "routing": {"parking_dist": 1},
        "storage_zones": [
            {
                "slms": [
                    {
                        "location": [0, 0],
                        "site_seperation": [6, 6],
                        "r": 1,
                        "c": 3,
                    }
                ]
            }
        ],
        "entanglement_zones": [
            {
                "slms": [
                    {
                        "location": [0, 12],
                        "site_seperation": [3, 6],
                        "r": 1,
                        "c": 2,
                    }
                ]
            }
        ],
    }


def test_internal_compile_matches_frozen_companion_trace():
    architecture = load_architecture("default")
    actual = compile_interactions(INTERACTIONS, architecture, atom_count=3)
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert actual.atom_count == expected["atom_count"]
    assert json.loads(json.dumps(actual.instructions)) == expected["instructions"]
    two_qubit_gates = [
        gate
        for instruction in actual.instructions
        if instruction.get("type") == "2qGate"
        for gate in instruction["gates"]
    ]
    assert two_qubit_gates
    assert all(type(gate) is tuple for gate in two_qubit_gates)
    assert exported_compile_interactions is compile_interactions


def test_internal_compile_has_no_file_or_console_side_effects(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    trace = compile_interactions(
        (ZapInteraction("na.0", (0, 1)),),
        _small_architecture(),
        atom_count=2,
    )

    assert trace.atom_count == 2
    assert not (tmp_path / "results").exists()
    assert capsys.readouterr() == ("", "")


def test_internal_compile_initializes_idle_atoms():
    trace = compile_interactions(
        (ZapInteraction("na.0", (0, 1)),),
        _small_architecture(),
        atom_count=3,
    )

    assert [location["id"] for location in trace.instructions[0]["locs"]] == [0, 1, 2]


def test_internal_compile_rejects_more_atoms_than_storage_traps():
    with pytest.raises(ValueError, match="3 atoms.*2 storage traps"):
        compile_interactions(
            (ZapInteraction("na.0", (0, 1)),),
            {
                **_small_architecture(),
                "storage_zones": [
                    {
                        "slms": [
                            {
                                "location": [0, 0],
                                "site_seperation": [6, 6],
                                "r": 1,
                                "c": 2,
                            }
                        ]
                    }
                ],
            },
            atom_count=3,
        )


def test_internal_compile_normalizes_integral_float_initial_mapping():
    trace = compile_interactions(
        (ZapInteraction("na.0", (0, 1)),),
        _small_architecture(),
        atom_count=2,
        initial_mapping=((0.0, 0.0), (6.0, 0.0)),
    )

    coordinates = [
        coordinate
        for location in trace.instructions[0]["locs"]
        for coordinate in (location["x"], location["y"])
    ]
    assert coordinates == [0, 0, 6, 0]
    assert all(type(coordinate) is int for coordinate in coordinates)


@pytest.mark.parametrize(
    "invalid_coordinate",
    (True, 0.5, float("nan"), float("inf")),
    ids=("bool", "fractional", "nan", "infinity"),
)
def test_internal_compile_rejects_non_integer_initial_mapping_coordinates(
    invalid_coordinate,
):
    with pytest.raises(
        ValueError, match="initial_mapping coordinates must be finite integers"
    ):
        compile_interactions(
            (ZapInteraction("na.0", (0, 1)),),
            _small_architecture(),
            atom_count=2,
            initial_mapping=((invalid_coordinate, 0), (6, 0)),
        )


@pytest.mark.parametrize("atom_count", (0, True))
def test_internal_compile_rejects_non_positive_exact_integer_atom_count(atom_count):
    with pytest.raises(ValueError, match="atom_count"):
        compile_interactions((), _small_architecture(), atom_count=atom_count)


def test_internal_compile_rejects_an_atom_outside_the_declared_count():
    with pytest.raises(ValueError, match="outside atom_count"):
        compile_interactions(
            (ZapInteraction("na.0", (0, 2)),),
            _small_architecture(),
            atom_count=2,
        )


def test_internal_compile_rejects_unsupported_scheduling_strategy():
    with pytest.raises(ValueError, match="only asap_joint"):
        compile_interactions(
            (),
            _small_architecture(),
            atom_count=1,
            scheduling_strategy="asap_separate",
        )


def test_internal_compile_rejects_duplicate_operation_ids():
    with pytest.raises(ValueError, match="operation_id values must be unique"):
        compile_interactions(
            (ZapInteraction("na.0", (0,)), ZapInteraction("na.0", (1,))),
            _small_architecture(),
            atom_count=2,
        )


def test_internal_compile_is_deterministic_across_two_calls():
    architecture = load_architecture("default")

    first = compile_interactions(INTERACTIONS, architecture, atom_count=3)
    second = compile_interactions(INTERACTIONS, architecture, atom_count=3)

    assert first == second


@pytest.mark.parametrize("atom_count", (20, 50))
def test_internal_compile_handles_large_disjoint_cz_layers(atom_count):
    interactions = tuple(
        ZapInteraction(f"cz.{first // 2}", (first, first + 1))
        for first in range(0, atom_count, 2)
    )

    trace = compile_interactions(
        interactions,
        load_architecture("default"),
        atom_count=atom_count,
    )

    operation_ids = [
        operation_id
        for instruction in trace.instructions
        for operation_id in instruction.get("operation_ids", ())
    ]
    assert trace.atom_count == atom_count
    assert operation_ids == [interaction.operation_id for interaction in interactions]
