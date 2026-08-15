"""Three-level atom portable-calibration value tests."""

from copy import deepcopy
from dataclasses import fields
from importlib import resources
import inspect
import json
from math import inf, nan, pi

import pytest

from fatqat.emulator._core.model_document import CalibrationIdentity, FormatIdentity
from fatqat.emulator.atom_3level.calibration import (
    Atom3LevelCalibration,
    _PARSERS,
    default_atom_3level_calibration,
)
from fatqat.errors import BackendValidationError


def test_calibration_is_direct_frozen_slotted_semantic_value(
    atom_3level_calibration_document,
):
    pristine = deepcopy(atom_3level_calibration_document)
    value = Atom3LevelCalibration(atom_3level_calibration_document)
    atom_3level_calibration_document["recipes"]["cz"]["phase_offset"] = 99
    assert tuple(inspect.signature(Atom3LevelCalibration).parameters) == ("document",)
    assert value == Atom3LevelCalibration(pristine)
    assert value.format == FormatIdentity("atom.rb87_rydberg_3level_fixed_pulse", 1)
    assert value.identity == CalibrationIdentity("rb87_53s_lukin_2023_v1", "2026-08-05")
    assert value.phase_offset_rad == -0.7318
    assert value.cz_duration_us == pytest.approx(2 * pi * 1.215 / 28.902652413026097)
    with pytest.raises(TypeError):
        hash(value)
    with pytest.raises(AttributeError):
        value.identity = object()
    for removed in ("_normalized" + "_recipes", "registry", "parser"):
        assert not hasattr(value, removed)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "format"),
        (("format",), "id"),
        (("calibration",), "id"),
        (("units",), "angle"),
        (("recipes",), "rx_ry"),
        (("recipes", "rx_ry"), "omega_01"),
        (("recipes", "cz"), "omega_1r"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_calibration_requires_exact_schema(
    atom_3level_calibration_document, path, key, mutation
):
    cursor = atom_3level_calibration_document
    for part in path:
        cursor = cursor[part]
    if mutation == "missing":
        del cursor[key]
    else:
        cursor["unexpected"] = None
    with pytest.raises(BackendValidationError):
        Atom3LevelCalibration(atom_3level_calibration_document)


@pytest.mark.parametrize(
    ("recipe", "field"),
    [("rx_ry", "omega_01"), ("cz", "omega_1r"), ("cz", "duration_area")],
)
@pytest.mark.parametrize("value", [True, 0, -1.0, inf, -inf, nan])
def test_required_recipe_scalars_are_positive_and_finite(
    atom_3level_calibration_document, recipe, field, value
):
    atom_3level_calibration_document["recipes"][recipe][field] = value
    with pytest.raises(BackendValidationError):
        Atom3LevelCalibration(atom_3level_calibration_document)


@pytest.mark.parametrize(
    "field",
    [
        "phase_amplitude",
        "phase_rate_ratio",
        "phase_offset",
        "linear_phase_rate_ratio",
        "local_z_correction",
    ],
)
@pytest.mark.parametrize("value", [True, inf, -inf, nan])
def test_phase_recipe_scalars_are_finite(
    atom_3level_calibration_document, field, value
):
    atom_3level_calibration_document["recipes"]["cz"][field] = value
    with pytest.raises(BackendValidationError):
        Atom3LevelCalibration(atom_3level_calibration_document)


def test_dispatch_precedes_body_validation(atom_3level_model_document):
    with pytest.raises(BackendValidationError, match="unknown format"):
        Atom3LevelCalibration(atom_3level_model_document)


def test_recipe_queries_preserve_public_values(atom_3level_calibration_document):
    value = Atom3LevelCalibration(atom_3level_calibration_document)
    assert value.recipe("rx_ry") == atom_3level_calibration_document["recipes"]["rx_ry"]
    assert value.recipe("cz") == atom_3level_calibration_document["recipes"]["cz"]
    with pytest.raises(BackendValidationError, match="unknown calibration recipe"):
        value.recipe("missing")


def test_default_calibration_is_fresh_and_loaded_from_package_resource():
    document = json.loads(
        resources.files("fatqat.emulator.atom_3level")
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    first = default_atom_3level_calibration()
    second = default_atom_3level_calibration()
    assert first is not second
    assert first == second == Atom3LevelCalibration(document)


def test_calibration_uses_one_immutable_exact_format_table():
    assert tuple(_PARSERS) == (
        FormatIdentity("atom.rb87_rydberg_3level_fixed_pulse", 1),
    )
    assert not next(
        field for field in fields(Atom3LevelCalibration) if field.name == "format"
    ).compare
    with pytest.raises(TypeError):
        _PARSERS[FormatIdentity("other", 1)] = object()
    with pytest.raises(TypeError):
        del _PARSERS[FormatIdentity("atom.rb87_rydberg_3level_fixed_pulse", 1)]
