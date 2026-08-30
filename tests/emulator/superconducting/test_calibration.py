"""Superconducting portable-calibration value tests."""

from copy import deepcopy
from importlib import resources
import inspect
import json
from math import inf, nan

import pytest

from fatqat.emulator.superconducting.calibration import (
    TransmonCalibration,
    default_transmon_calibration,
)
from fatqat.errors import BackendValidationError


def test_calibration_is_direct_frozen_slotted_semantic_value(calibration_document):
    pristine = deepcopy(calibration_document)
    value = TransmonCalibration(calibration_document)
    calibration_document["recipes"]["rx_ry"]["duration"] = 99
    calibration_document["recipes"]["cz"]["edges"][0]["recipe"]["duration"] = 99
    assert tuple(inspect.signature(TransmonCalibration).parameters) == ("document",)
    assert value == TransmonCalibration(pristine)
    changed_identity = deepcopy(pristine)
    changed_identity["calibration"]["revision"] = "different"
    assert value != TransmonCalibration(changed_identity)
    assert value._rx_ry_duration_ns == 20.0
    assert value._cz_recipe("q0", "q1").duration_ns == 60.0
    with pytest.raises(TypeError):
        hash(value)
    assert not hasattr(value, "format")
    assert not hasattr(value, "identity")
    for removed in (
        "_normalized" + "_recipes",
        "_cz_default",
        "_cz_overrides",
        "registry",
        "parser",
    ):
        assert not hasattr(value, removed)


def test_calibration_copies_edge_containers_and_ignores_list_order(
    calibration_document,
):
    second = deepcopy(calibration_document["recipes"]["cz"]["edges"][0])
    second["canonical_edge"] = ["q2", "q3"]
    second["recipe"]["detuned_subsystem"] = "q3"
    calibration_document["recipes"]["cz"]["edges"].append(second)
    reversed_document = deepcopy(calibration_document)
    reversed_document["recipes"]["cz"]["edges"].reverse()
    value = TransmonCalibration(calibration_document)
    assert value == TransmonCalibration(reversed_document)

    calibration_document["recipes"]["cz"]["edges"][0]["canonical_edge"][0] = "changed"
    calibration_document["recipes"]["cz"]["edges"].clear()
    assert value._cz_recipe("q0", "q1").duration_ns == 60.0
    assert value._cz_recipe("q3", "q2").detuned_subsystem == "q3"


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "format"),
        (("format",), "id"),
        (("calibration",), "id"),
        (("units",), "time"),
        (("recipes",), "rx_ry"),
        (("recipes", "rx_ry"), "duration"),
        (("recipes", "iswap"), "duration"),
        (("recipes", "cz"), "edges"),
        (("recipes", "cz", "edges", 0), "canonical_edge"),
        (("recipes", "cz", "edges", 0, "recipe"), "duration"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_calibration_requires_exact_schema(calibration_document, path, key, mutation):
    cursor = calibration_document
    for part in path:
        cursor = cursor[part]
    if mutation == "missing":
        del cursor[key]
    else:
        cursor["unexpected"] = None
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


def test_calibration_rejects_wrong_units_and_model_field(calibration_document):
    calibration_document["units"]["frequency"] = "MHz"
    with pytest.raises(BackendValidationError, match="supported calibration units"):
        TransmonCalibration(calibration_document)


def test_calibration_accepts_only_exact_generated_provenance(calibration_document):
    calibration_document["provenance"] = {
        "kind": "generated_reference_recipe",
        "generator_version": 1,
        "numerically_calibrated": False,
    }
    assert TransmonCalibration(calibration_document)._cz_recipe("q0", "q1")


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {
            "kind": "generated_reference_recipe",
            "generator_version": 1,
            "numerically_calibrated": False,
            "unexpected": None,
        },
        {
            "kind": "other",
            "generator_version": 1,
            "numerically_calibrated": False,
        },
        {
            "kind": "generated_reference_recipe",
            "generator_version": True,
            "numerically_calibrated": False,
        },
        {
            "kind": "generated_reference_recipe",
            "generator_version": 2,
            "numerically_calibrated": False,
        },
        {
            "kind": "generated_reference_recipe",
            "generator_version": 1,
            "numerically_calibrated": True,
        },
        {
            "kind": "generated_reference_recipe",
            "generator_version": 1,
            "numerically_calibrated": 0,
        },
    ],
)
def test_calibration_rejects_other_provenance(calibration_document, provenance):
    calibration_document["provenance"] = provenance
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


def test_calibration_rejects_legacy_cz_defaults_and_overrides(calibration_document):
    calibration_document["recipes"]["cz"] = {"default": {}, "overrides": []}
    with pytest.raises(BackendValidationError, match="missing.*edges.*unknown"):
        TransmonCalibration(calibration_document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", 0),
        ("duration", inf),
        ("duration", nan),
        ("ramp_duration", -1),
        ("ramp_duration", inf),
        ("park_detuning_ghz", inf),
        ("branch_tolerance_ghz", -1),
        ("branch_tolerance_ghz", inf),
    ],
)
def test_cz_numbers_are_validated(calibration_document, field, value):
    calibration_document["recipes"]["cz"]["edges"][0]["recipe"][field] = value
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


def test_cz_park_and_absolute_edge_lookup_are_validated(calibration_document):
    recipe = calibration_document["recipes"]["cz"]["edges"][0]["recipe"]
    recipe["ramp_duration"] = 30
    with pytest.raises(BackendValidationError, match="inconsistent"):
        TransmonCalibration(calibration_document)
    recipe.update(
        {
            "detuned_subsystem": "q1",
            "duration": 64,
            "ramp_duration": 5,
            "park_detuning_ghz": 0.24,
        }
    )
    value = TransmonCalibration(calibration_document)
    forward = value._cz_recipe("q0", "q1")
    reverse = value._cz_recipe("q1", "q0")
    assert forward is reverse
    assert forward.detuned_subsystem == "q1"
    assert forward.duration_ns == 64
    assert forward.park_detuning_ghz == 0.24
    assert value._cz_recipe("q0", "q2") is None


@pytest.mark.parametrize(
    "endpoints",
    [
        [],
        ["q0"],
        ["q0", "q0"],
        ["q0", "q1", "q2"],
        ["q0", 1],
        ["q1", "q0"],
        ["q9", "q10"],
    ],
)
def test_cz_edges_are_distinct_canonical_string_pairs(calibration_document, endpoints):
    calibration_document["recipes"]["cz"]["edges"][0]["canonical_edge"] = endpoints
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


@pytest.mark.parametrize("detuned_subsystem", ["", "q2", 0])
def test_cz_detuned_subsystem_must_be_an_endpoint(
    calibration_document, detuned_subsystem
):
    calibration_document["recipes"]["cz"]["edges"][0]["recipe"][
        "detuned_subsystem"
    ] = detuned_subsystem
    with pytest.raises(BackendValidationError, match="detuned_subsystem"):
        TransmonCalibration(calibration_document)


def test_cz_edges_are_unique_and_may_be_empty(calibration_document):
    edges = calibration_document["recipes"]["cz"]["edges"]
    duplicate = deepcopy(edges[0])
    edges.append(duplicate)
    with pytest.raises(BackendValidationError, match="duplicates a canonical"):
        TransmonCalibration(calibration_document)

    edges.clear()
    calibration = TransmonCalibration(calibration_document)
    assert calibration._cz_recipe("q0", "q1") is None


def test_default_calibration_is_fresh_and_loaded_from_resource():
    document = json.loads(
        resources.files("fatqat.emulator.superconducting")
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    first = default_transmon_calibration()
    second = default_transmon_calibration()
    assert first is not second
    assert first == second == TransmonCalibration(document)
