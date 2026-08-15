"""Shared identities and exact-format document dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .document_validation import _exact_keys, _fail, _mapping, _string, _version


@dataclass(frozen=True, slots=True)
class FormatIdentity:
    """Public identity of a persisted document format."""

    id: str
    version: int

    def __post_init__(self) -> None:
        _string(self.id, "format.id")
        _version(self.version, "format.version")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Durable identity of one model parameter snapshot."""

    id: str
    revision: str

    def __post_init__(self) -> None:
        _string(self.id, "model.id")
        _string(self.revision, "model.revision")


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    """Durable identity of one calibration snapshot."""

    id: str
    revision: str

    def __post_init__(self) -> None:
        _string(self.id, "calibration.id")
        _string(self.revision, "calibration.revision")


def _parse_format_identity(value: Any, path: str) -> FormatIdentity:
    """Parse an exact ``{id, version}`` format identity."""
    data = _mapping(value, path)
    _exact_keys(data, {"id", "version"}, path)
    return FormatIdentity(
        _string(data["id"], f"{path}.id"),
        _version(data["version"], f"{path}.version"),
    )


def _parse_model_identity(value: Any, path: str) -> ModelIdentity:
    """Parse an exact ``{id, revision}`` model identity."""
    data = _mapping(value, path)
    _exact_keys(data, {"id", "revision"}, path)
    return ModelIdentity(
        _string(data["id"], f"{path}.id"),
        _string(data["revision"], f"{path}.revision"),
    )


def _parse_calibration_identity(value: Any, path: str) -> CalibrationIdentity:
    """Parse an exact ``{id, revision}`` calibration identity."""
    data = _mapping(value, path)
    _exact_keys(data, {"id", "revision"}, path)
    return CalibrationIdentity(
        _string(data["id"], f"{path}.id"),
        _string(data["revision"], f"{path}.revision"),
    )


def _dispatch_document(
    document: Any,
    path: str,
    parsers: Mapping[FormatIdentity, Callable[[Mapping[str, Any]], Any]],
) -> tuple[FormatIdentity, Any]:
    """Select one exact parser without copying or recursively prechecking data."""
    data = _mapping(document, path)
    if "format" not in data:
        _fail(path, "missing ['format']")
    identity = _parse_format_identity(data["format"], f"{path}.format")
    parser = parsers.get(identity)
    if parser is None:
        _fail("format", f"unknown format {identity.id} version {identity.version}")
    return identity, parser(data)


__all__ = ["FormatIdentity", "ModelIdentity", "CalibrationIdentity"]
