"""Public reference-model catalog contract."""

from copy import deepcopy

import pytest

import fatqat as fq

_CATALOG_MODELS = {
    "atom2level.reference": fq.emulator.Atom2LevelModel,
    "atom3level.reference": fq.emulator.Atom3LevelModel,
    "transmon.reference": fq.emulator.TransmonModel,
}


def test_available_model_documents_is_exact_and_deterministic():
    assert fq.emulator.available_model_documents() == tuple(_CATALOG_MODELS)


def test_load_model_document_rejects_non_string_name():
    with pytest.raises(TypeError):
        fq.emulator.load_model_document(None)


def test_load_model_document_rejects_unknown_name():
    with pytest.raises(KeyError):
        fq.emulator.load_model_document("missing.reference")


def test_load_model_document_returns_independent_mutable_graphs():
    first = fq.emulator.load_model_document("atom3level.reference")
    original = deepcopy(first)
    first["parameters"]["mass"] = 1.0

    second = fq.emulator.load_model_document("atom3level.reference")
    assert second == original


@pytest.mark.parametrize(
    ("name", "model_type"),
    _CATALOG_MODELS.items(),
)
def test_reference_document_constructs_its_model(name, model_type):
    document = fq.emulator.load_model_document(name)

    assert isinstance(model_type.from_document(document), model_type)


def test_atom_reference_documents_share_the_53s_physical_profile():
    atom2 = fq.emulator.load_model_document("atom2level.reference")
    atom3 = fq.emulator.load_model_document("atom3level.reference")

    assert atom2["system"]["species"] == atom3["system"]["species"]
    assert atom2["system"]["basis"]["g"] == atom3["system"]["basis"]["1"]
    assert atom2["system"]["basis"]["r"] == atom3["system"]["basis"]["r"]
    assert atom2["parameters"]["c6"] == atom3["parameters"]["c6"]
    assert atom2["references"] == atom3["references"]
