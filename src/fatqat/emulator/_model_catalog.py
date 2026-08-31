"""Private resource loading for the public reference-model catalog."""

from __future__ import annotations

from importlib.resources import files
import json

from . import _model_documents

_DOCUMENT_RESOURCES = {
    "atom2level.reference": "atom2level_reference.json",
    "transmon.reference": "transmon_reference.json",
    "transmon.single": "transmon_single.json",
}
_DOCUMENT_NAMES = tuple(_DOCUMENT_RESOURCES)


def available_model_documents() -> tuple[str, ...]:
    """Return the stable names of package-shipped reference model documents.

    Returns:
        Catalog names in deterministic order. Each name identifies a reference
        snapshot, not a universal physical default.
    """
    return _DOCUMENT_NAMES


def load_model_document(name: str) -> dict[str, object]:
    """Load a fresh mutable copy of one reference model document.

    Inspect the returned identity, units, parameters, and references before
    passing it to the corresponding model family's ``from_document`` method.
    Mutating a returned document never changes later loads.

    Args:
        name: One name returned by ``available_model_documents()``.

    Returns:
        A newly decoded, JSON-compatible mutable dictionary.

    Raises:
        TypeError: If ``name`` is not a string.
        KeyError: If ``name`` is not in the public catalog.
    """
    if not isinstance(name, str):
        raise TypeError("model document name must be a string")
    try:
        resource_name = _DOCUMENT_RESOURCES[name]
    except KeyError:
        available = ", ".join(_DOCUMENT_NAMES)
        raise KeyError(
            f"unknown model document {name!r}; available names: {available}"
        ) from None
    text = files(_model_documents).joinpath(resource_name).read_text(encoding="utf-8")
    document = json.loads(text)
    assert isinstance(document, dict)
    return document


__all__ = ["available_model_documents", "load_model_document"]
