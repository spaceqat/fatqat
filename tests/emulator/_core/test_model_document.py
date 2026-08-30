"""Shared exact-envelope validation for model documents."""

from __future__ import annotations

import pytest

from fatqat.emulator._core.model_document import _validate_model_document_envelope
from fatqat.errors import BackendValidationError


@pytest.mark.parametrize(
    "references",
    [["doi:1"], []],
)
def test_model_envelope_accepts_base_schema_and_valid_references(references):
    base = {
        "format": {},
        "model": {},
        "system": {},
        "units": {},
        "parameters": {},
    }
    _validate_model_document_envelope(base, "physics model")
    _validate_model_document_envelope(
        {**base, "references": references}, "physics model"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.update(extra=None),
        lambda document: document.pop("system"),
        lambda document: document.update(references="doi:1"),
        lambda document: document.update(references=[""]),
        lambda document: document.update(
            provenance={"description": "obsolete", "sources": []}
        ),
    ],
)
def test_model_envelope_rejects_nonexact_or_invalid_references(mutate):
    document = {
        "format": {},
        "model": {},
        "system": {},
        "units": {},
        "parameters": {},
    }
    mutate(document)
    with pytest.raises(BackendValidationError):
        _validate_model_document_envelope(document, "physics model")
