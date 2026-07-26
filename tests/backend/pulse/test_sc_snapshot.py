"""Persistence and calibration checks for the SC transmon/exchange model."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fatqat.backends.pulse.superconducting import (
    PhysicsModelSpec,
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendValidationError

_FIXTURES = Path(__file__).parent / "fixtures"


def _model_document():
    return json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())


def _calibration_document():
    return json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.update(format="foreign.physics-model"),
        lambda document: document.update(schema_version=2),
        lambda document: document["builder"].update(version=99),
        lambda document: document.update(executable="import this"),
        lambda document: document["parameters"]["subsystems"][0].update(
            frequency=float("nan")
        ),
        lambda document: document["parameters"]["subsystems"].append(
            {"id": "q0", "frequency": 4.9, "anharmonicity": -0.3}
        ),
        lambda document: document["parameters"]["couplings"][0].update(
            subsystems=["q0", "missing"]
        ),
    ],
)
def test_snapshot_loader_rejects_invalid_or_non_data_documents(mutate):
    document = _model_document()
    mutate(document)
    with pytest.raises(BackendValidationError):
        load_physics_model(document)


def test_spec_is_data_only_and_resolves_through_the_trusted_registry():
    document = _model_document()
    assert PhysicsModelSpec.from_mapping(document).model.id == "test-sc-2q"
    assert load_physics_model(document).subsystem_ids == ("q0", "q1")
    document["parameters"]["callback"] = lambda: None
    with pytest.raises(BackendValidationError, match="JSON data"):
        PhysicsModelSpec.from_mapping(document)


def test_calibration_is_separate_and_exactly_identity_bound():
    model = load_physics_model(_model_document())
    calibration = load_calibration_spec(_calibration_document(), model)
    assert calibration.key == model.key
    assert calibration.recipe("rx_ry")["duration_ns"] == 20.0

    invalid = deepcopy(_calibration_document())
    invalid["model"]["revision"] = "different"
    with pytest.raises(BackendValidationError, match="identity"):
        load_calibration_spec(invalid, model)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["recipes"]["rx_ry"].update(duration_ns=0),
        lambda document: document["recipes"]["cz"]["edges"][0].update(
            detuning_subsystem="missing"
        ),
        lambda document: document["recipes"]["cz"]["edges"][0].pop(
            "phase_corrections_rad"
        ),
    ],
)
def test_calibration_rejects_incomplete_or_invalid_recipe_values(mutate):
    document = _calibration_document()
    mutate(document)
    with pytest.raises(BackendValidationError):
        load_calibration_spec(document, load_physics_model(_model_document()))


def test_calibration_rejects_an_rz_recipe_including_an_arbitrary_scale():
    document = _calibration_document()
    document["recipes"]["rz"] = {"frame_scale": 2.0}
    with pytest.raises(BackendValidationError, match="unknown"):
        load_calibration_spec(document, load_physics_model(_model_document()))


def test_calibration_permits_unreferenced_uncalibrated_model_edges():
    model_document = _model_document()
    model_document["parameters"]["subsystems"].append(
        {"id": "q2", "frequency": 5.35, "anharmonicity": -0.23}
    )
    model_document["parameters"]["couplings"].append(
        {"id": "e1", "subsystems": ["q1", "q2"]}
    )
    model = load_physics_model(model_document)

    calibration = load_calibration_spec(_calibration_document(), model)
    assert len(calibration.recipe("cz")["edges"]) == 1


@pytest.mark.parametrize("drag_coefficient", [0.0, -0.5])
def test_calibration_permits_dimensionless_drag_sign_or_disable(drag_coefficient):
    document = _calibration_document()
    document["recipes"]["rx_ry"]["drag_coefficient"] = drag_coefficient
    calibration = load_calibration_spec(document, load_physics_model(_model_document()))
    assert calibration.recipe("rx_ry")["drag_coefficient"] == drag_coefficient
