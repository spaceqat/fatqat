"""Shared document identities and JSON-only construction boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from fatqat.emulator._core.model_document import (
    CalibrationIdentity,
    FormatIdentity,
    ModelIdentity,
    _dispatch_document,
    _parse_calibration_identity,
    _parse_format_identity,
    _parse_model_identity,
    _validate_model_document_envelope,
)
from fatqat.errors import BackendValidationError


def test_identity_values_are_frozen_and_compare_by_value():
    assert FormatIdentity("example", 1) == FormatIdentity("example", 1)
    assert ModelIdentity("model", "revision") == ModelIdentity("model", "revision")
    assert CalibrationIdentity("cal", "revision") == CalibrationIdentity(
        "cal", "revision"
    )

    with pytest.raises(FrozenInstanceError):
        FormatIdentity("example", 1).id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "arguments", "path"),
    [
        (FormatIdentity, ("", 1), "format.id"),
        (FormatIdentity, ("example", True), "format.version"),
        (FormatIdentity, ("example", 0), "format.version"),
        (ModelIdentity, ("", "revision"), "model.id"),
        (ModelIdentity, ("model", ""), "model.revision"),
        (CalibrationIdentity, ("", "revision"), "calibration.id"),
        (CalibrationIdentity, ("cal", ""), "calibration.revision"),
    ],
)
def test_invalid_identity_values_are_validation_errors(factory, arguments, path):
    with pytest.raises(BackendValidationError, match=path):
        factory(*arguments)


def test_identity_parsers_require_exact_keys_and_qualified_paths():
    assert _parse_format_identity(
        {"id": "example", "version": 1}, "document.format"
    ) == FormatIdentity("example", 1)
    assert _parse_model_identity(
        {"id": "model", "revision": "r1"}, "document.model"
    ) == ModelIdentity("model", "r1")
    assert _parse_calibration_identity(
        {"id": "cal", "revision": "r1"}, "document.calibration"
    ) == CalibrationIdentity("cal", "r1")

    with pytest.raises(BackendValidationError, match=r"document\.format:.*unknown"):
        _parse_format_identity(
            {"id": "example", "version": 1, "extra": None}, "document.format"
        )
    with pytest.raises(BackendValidationError, match=r"document\.model:.*missing"):
        _parse_model_identity({"id": "model"}, "document.model")


def test_exact_format_dispatch_selects_one_parser_and_preserves_parser_paths():
    calls = []

    def parse(document):
        calls.append(document)
        raise BackendValidationError("physics model.parameters.value: invalid")

    parsers = MappingProxyType({FormatIdentity("example", 1): parse})
    document = {"format": {"id": "example", "version": 1}}
    with pytest.raises(BackendValidationError, match=r"parameters\.value"):
        _dispatch_document(document, "physics model", parsers)
    assert calls == [document]

    document["format"]["version"] = 2
    with pytest.raises(BackendValidationError, match="unknown format"):
        _dispatch_document(document, "physics model", parsers)


def test_booleans_are_rejected_by_exact_integer_and_numeric_validators():
    with pytest.raises(BackendValidationError, match="format.version"):
        _parse_format_identity({"id": "example", "version": True}, "format")


@pytest.mark.parametrize(
    "provenance",
    [
        {"description": "Literature-derived reference.", "sources": ["doi:1"]},
        {"description": "Synthetic reference.", "sources": []},
    ],
)
def test_model_envelope_accepts_base_schema_and_valid_provenance(provenance):
    base = {
        "format": {},
        "model": {},
        "system": {},
        "units": {},
        "parameters": {},
    }
    _validate_model_document_envelope(base, "physics model")
    _validate_model_document_envelope(
        {**base, "provenance": provenance}, "physics model"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.update(extra=None),
        lambda document: document.pop("system"),
        lambda document: document.update(provenance={"description": "text"}),
        lambda document: document.update(provenance={"description": "", "sources": []}),
        lambda document: document.update(
            provenance={"description": "text", "sources": "citation"}
        ),
        lambda document: document.update(
            provenance={"description": "text", "sources": [""]}
        ),
        lambda document: document.update(
            provenance={"description": "text", "sources": [], "extra": None}
        ),
    ],
)
def test_model_envelope_rejects_nonexact_or_invalid_provenance(mutate):
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
