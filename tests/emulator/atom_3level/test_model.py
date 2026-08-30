"""Three-level atom physics-model value tests."""

from copy import deepcopy
import inspect
from math import nan

import pytest

from fatqat.emulator._core.target import _ControlAddress, _FrameAddress
from fatqat.emulator.atom_3level.model import Atom3LevelModel
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
    assert first.control.raman(2) == second.control.raman(2)
    assert first.frame(2) == second.frame(2)
    with pytest.raises(BackendValidationError):
        first.control.raman(-1)
    with pytest.raises(BackendValidationError):
        first.control.rydberg(True)
