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
    calibration_document["recipes"]["cz"]["overrides"][0]["recipe"]["duration"] = 99
    assert tuple(inspect.signature(TransmonCalibration).parameters) == ("document",)
    assert value == TransmonCalibration(pristine)
    changed_identity = deepcopy(pristine)
    changed_identity["calibration"]["revision"] = "different"
    assert value != TransmonCalibration(changed_identity)
    assert value._rx_ry_duration_ns == 20.0
    assert value._cz_duration_ns("q0", "q1") == 60.0
    with pytest.raises(TypeError):
        hash(value)
    assert not hasattr(value, "format")
    assert not hasattr(value, "identity")
    for removed in ("_normalized" + "_recipes", "registry", "parser"):
        assert not hasattr(value, removed)


def test_calibration_copies_override_containers_and_ignores_list_order(
    calibration_document,
):
    second = deepcopy(calibration_document["recipes"]["cz"]["overrides"][0])
    second["device_operands"] = ["q2", "q3"]
    calibration_document["recipes"]["cz"]["overrides"].append(second)
    reversed_document = deepcopy(calibration_document)
    reversed_document["recipes"]["cz"]["overrides"].reverse()
    value = TransmonCalibration(calibration_document)
    assert value == TransmonCalibration(reversed_document)

    calibration_document["recipes"]["cz"]["overrides"][0]["device_operands"][
        0
    ] = "changed"
    calibration_document["recipes"]["cz"]["overrides"].clear()
    assert value._cz_duration_ns("q0", "q1") == 60.0
    assert value._cz_duration_ns("q2", "q3") == 60.0


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
        (("recipes", "cz"), "default"),
        (("recipes", "cz", "default"), "detuning_operand"),
        (("recipes", "cz", "overrides", 0), "device_operands"),
        (("recipes", "cz", "overrides", 0, "recipe"), "duration"),
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


@pytest.mark.parametrize("operand", [True, -1, 2, 0.0, "0"])
def test_cz_detuning_operand_is_binary_integer(calibration_document, operand):
    calibration_document["recipes"]["cz"]["default"]["detuning_operand"] = operand
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", 0),
        ("duration", inf),
        ("duration", nan),
        ("ramp_duration", -1),
        ("ramp_duration", inf),
        ("detuning", inf),
    ],
)
def test_cz_numbers_are_validated(calibration_document, field, value):
    calibration_document["recipes"]["cz"]["default"][field] = value
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


def test_cz_park_and_ordered_overrides_are_validated(calibration_document):
    calibration_document["recipes"]["cz"]["default"]["ramp_duration"] = 30
    with pytest.raises(BackendValidationError, match="inconsistent"):
        TransmonCalibration(calibration_document)
    calibration_document["recipes"]["cz"]["default"]["ramp_duration"] = 5
    override = calibration_document["recipes"]["cz"]["overrides"][0]
    override["recipe"]["duration"] = 64
    override["recipe"]["detuning_operand"] = 1
    value = TransmonCalibration(calibration_document)
    assert value._cz_duration_ns("q0", "q1") == 64
    assert value._cz_detuning_subsystem("q0", "q1") == "q1"
    assert value._cz_duration_ns("q1", "q0") == 60
    calibration_document["recipes"]["cz"]["overrides"].append(deepcopy(override))
    with pytest.raises(BackendValidationError, match="duplicates an ordered"):
        TransmonCalibration(calibration_document)


@pytest.mark.parametrize(
    "operands", [[], ["q0"], ["q0", "q0"], ["q0", "q1", "q2"], ["q0", 1]]
)
def test_cz_override_operands_are_two_distinct_strings(calibration_document, operands):
    calibration_document["recipes"]["cz"]["overrides"][0]["device_operands"] = operands
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


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
