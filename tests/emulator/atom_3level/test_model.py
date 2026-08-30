"""Three-level atom physics-model value tests."""

from copy import deepcopy
from importlib.resources import files
import inspect
import json
from math import nan

import pytest

from fatqat.emulator import ControlChannel, PhaseShift, load_model_document
from fatqat.emulator import _model_documents
from fatqat.emulator._core.target import _ControlAddress, _FrameAddress
from fatqat.emulator._atom_3level.model import Atom3LevelModel
from fatqat.errors import BackendValidationError


def test_model_is_direct_frozen_slotted_semantic_value(atom_3level_model_document):
    pristine = deepcopy(atom_3level_model_document)
    model = Atom3LevelModel.from_document(atom_3level_model_document)
    atom_3level_model_document["parameters"]["c6"] = -1

    assert tuple(inspect.signature(Atom3LevelModel.from_document).parameters) == (
        "document",
    )
    assert model == Atom3LevelModel.from_document(pristine)
    assert isinstance(model == Atom3LevelModel.from_document(pristine), bool)
    changed_identity = deepcopy(pristine)
    changed_identity["model"]["revision"] = "different"
    assert model != Atom3LevelModel.from_document(changed_identity)
    assert not hasattr(model, "format")
    assert not hasattr(model, "identity")
    assert model.kind == "atom.rydberg_3level"
    assert model.species == "Rb87"
    assert model.local_dimension == 3
    assert model.computational_states == {
        "0": "5S1/2,F=1,mF=0",
        "1": "5S1/2,F=2,mF=0",
    }
    assert model.c6_angular_per_us_um6 == 180955.73684677208
    assert (model.mass_unit, model.distance_unit, model.time_unit) == ("u", "um", "us")
    with pytest.raises(TypeError):
        hash(model)
    for removed in (
        "_normalized" + "_state",
        "_model" + "_key",
        "registry",
        "parser",
    ):
        assert not hasattr(model, removed)


def test_direct_construction_is_removed(atom_3level_model_document):
    with pytest.raises(TypeError, match="from_document"):
        Atom3LevelModel()
    with pytest.raises(TypeError, match="from_document"):
        Atom3LevelModel(atom_3level_model_document)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "format"),
        (("format",), "id"),
        (("model",), "id"),
        (("system",), "species"),
        (("system", "basis"), "0"),
        (("system", "transitions", "rydberg"), "from"),
        (("units",), "mass"),
        (("parameters",), "mass"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_model_requires_exact_schema(atom_3level_model_document, path, key, mutation):
    cursor = atom_3level_model_document
    for part in path:
        cursor = cursor[part]
    if mutation == "missing":
        del cursor[key]
    else:
        cursor["unexpected"] = None
    with pytest.raises(BackendValidationError):
        Atom3LevelModel.from_document(atom_3level_model_document)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["format"].update(id="foreign"),
        lambda d: d["format"].update(version=True),
        lambda d: d["system"].update(species="Cs133"),
        lambda d: d["system"]["basis"].update(**{"0": "other"}),
        lambda d: d["system"]["transitions"]["rydberg"].update(**{"from": "0"}),
        lambda d: d["units"].update(time="ns"),
        lambda d: d["parameters"].update(mass=0),
        lambda d: d["parameters"].update(c6=0),
        lambda d: d["parameters"].update(c6=nan),
        lambda d: d.update(callback=lambda: None),
    ],
)
def test_model_rejects_invalid_constants_and_numbers(
    atom_3level_model_document, mutate
):
    mutate(atom_3level_model_document)
    with pytest.raises(BackendValidationError):
        Atom3LevelModel.from_document(atom_3level_model_document)


def test_model_factories_return_portable_structural_addresses(
    atom_3level_model_document,
):
    first = Atom3LevelModel.from_document(atom_3level_model_document)
    second = Atom3LevelModel.from_document(deepcopy(atom_3level_model_document))
    assert isinstance(first.control.raman(2), _ControlAddress)
    assert isinstance(first.control.rydberg(2), _ControlAddress)
    assert isinstance(first.frame(2), _FrameAddress)
    assert isinstance(first.control.raman(2), ControlChannel)
    assert isinstance(first.control.rydberg(2), ControlChannel)
    assert first.control.raman(2) == second.control.raman(2)
    assert first.frame(2) == second.frame(2)
    assert PhaseShift(first.frame(2), 0.1).frame == first.frame(2)
    expected = {
        "raman": ("local", ("site",), "complex", "rad/us"),
        "rydberg": ("local", ("site",), "complex", "rad/us"),
    }
    assert tuple(first.available_controls) == tuple(expected)
    for name, metadata in expected.items():
        selector = first.available_controls[name]
        assert selector is getattr(first.control, name)
        assert (
            selector.scope,
            selector.operands,
            selector.coefficient_domain,
            selector.coefficient_unit,
        ) == metadata
    with pytest.raises(TypeError):
        first.available_controls["other"] = object()
    with pytest.raises(BackendValidationError):
        first.control.raman(-1)
    with pytest.raises(BackendValidationError):
        first.control.rydberg(True)


def test_internal_atom3_reference_matches_the_public_atom2_physical_profile():
    atom2 = load_model_document("atom2level.reference")
    atom3 = json.loads(
        files(_model_documents)
        .joinpath("atom3level_reference.json")
        .read_text(encoding="utf-8")
    )
    parsed = Atom3LevelModel.from_document(atom3)

    assert isinstance(parsed, Atom3LevelModel)
    assert atom2["system"]["species"] == atom3["system"]["species"]
    assert atom2["system"]["basis"]["g"] == atom3["system"]["basis"]["1"]
    assert atom2["system"]["basis"]["r"] == atom3["system"]["basis"]["r"]
    assert atom2["parameters"]["c6"] == atom3["parameters"]["c6"]
    assert atom2["references"] == atom3["references"]
