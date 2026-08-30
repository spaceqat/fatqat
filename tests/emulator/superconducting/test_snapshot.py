"""Durable transmon model and portable-calibration snapshot identities."""

from copy import deepcopy
import inspect

import pytest

from fatqat.emulator.superconducting import TransmonCalibration, TransmonModel
from fatqat.errors import BackendValidationError


def test_public_snapshot_constructors_are_document_only(
    model_document, calibration_document
):
    assert tuple(inspect.signature(TransmonModel.from_document).parameters) == (
        "document",
    )
    assert tuple(inspect.signature(TransmonCalibration).parameters) == ("document",)
    model = TransmonModel.from_document(model_document)
    calibration = TransmonCalibration(calibration_document)
    assert not hasattr(model, "format")
    assert not hasattr(model, "identity")
    assert not hasattr(calibration, "format")
    assert not hasattr(calibration, "identity")


def test_calibration_document_rejects_model_binding_fields(calibration_document):
    calibration_document["model"] = {
        "kind": "sc.transmon",
        "id": "test-sc-2q",
        "revision": "2026-07-26",
    }
    with pytest.raises(BackendValidationError):
        TransmonCalibration(calibration_document)


def test_portable_calibration_accepts_unused_ordered_override(calibration_document):
    document = deepcopy(calibration_document)
    document["recipes"]["cz"]["overrides"].append(
        {
            "device_operands": ["future-q0", "future-q1"],
            "recipe": deepcopy(document["recipes"]["cz"]["default"]),
        }
    )
    calibration = TransmonCalibration(document)
    assert calibration._cz_duration_ns("future-q0", "future-q1") == 60.0


def test_model_and_calibration_format_dispatch_remain_distinct(
    model_document, calibration_document
):
    with pytest.raises(BackendValidationError, match="unknown format"):
        TransmonModel.from_document(calibration_document)
    with pytest.raises(BackendValidationError, match="unknown format"):
        TransmonCalibration(model_document)


def test_document_identities_remain_validated_with_qualified_paths(
    model_document, calibration_document
):
    model_document["model"]["id"] = ""
    with pytest.raises(BackendValidationError, match=r"physics model\.model\.id"):
        TransmonModel.from_document(model_document)

    calibration_document["calibration"]["revision"] = ""
    with pytest.raises(
        BackendValidationError, match=r"calibration\.calibration\.revision"
    ):
        TransmonCalibration(calibration_document)
