"""Public reference-model catalog contract."""

from copy import deepcopy

import pytest

import fatqat as fq

_CATALOG_MODELS = {
    "atom2level.reference": fq.emulator.Atom2LevelModel,
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
    first = fq.emulator.load_model_document("atom2level.reference")
    original = deepcopy(first)
    first["parameters"]["c6"] = 1.0

    second = fq.emulator.load_model_document("atom2level.reference")
    assert second == original


@pytest.mark.parametrize(
    ("name", "model_type"),
    _CATALOG_MODELS.items(),
)
def test_reference_document_constructs_its_model(name, model_type):
    document = fq.emulator.load_model_document(name)

    assert isinstance(model_type.from_document(document), model_type)
