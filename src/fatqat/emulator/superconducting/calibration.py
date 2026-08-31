"""Fixed-pulse calibration values for superconducting transmons."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources as package_resources
from types import MappingProxyType
from typing import Any, ClassVar

from .._core.document_validation import _exact_keys, _fail, _mapping, _number, _string
from .._core.model_document import (
    _CalibrationIdentity,
    _FormatIdentity,
    _dispatch_document,
    _parse_calibration_identity,
)

_FORMAT = _FormatIdentity("sc.transmon_exchange_fixed_pulse", 1)
_UNITS = {"time": "ns", "frequency": "GHz", "dimensionless": "1"}


@dataclass(frozen=True, slots=True)
class _CzRecipe:
    detuned_subsystem: str
    duration_ns: float
    ramp_duration_ns: float
    park_detuning_ghz: float
    branch_tolerance_ghz: float


def _validate_single_recipe(
    value: Any,
    path: str,
    expected: set[str],
    *,
    positive_fields: set[str] | None = None,
) -> dict[str, float]:
    data = _mapping(value, path)
    _exact_keys(data, expected, path)
    positive_fields = expected if positive_fields is None else positive_fields
    return {
        name: _number(data[name], f"{path}.{name}", positive=name in positive_fields)
        for name in expected
    }


def _validate_cz_recipe(
    value: Any, path: str, canonical_edge: tuple[str, str]
) -> _CzRecipe:
    recipe = _mapping(value, path)
    _exact_keys(
        recipe,
        {
            "detuned_subsystem",
            "duration",
            "ramp_duration",
            "park_detuning_ghz",
            "branch_tolerance_ghz",
        },
        path,
    )
    detuned_subsystem = _string(
        recipe["detuned_subsystem"], f"{path}.detuned_subsystem"
    )
    if detuned_subsystem not in canonical_edge:
        _fail(
            f"{path}.detuned_subsystem",
            "must name one endpoint of the canonical edge",
        )
    duration = _number(recipe["duration"], f"{path}.duration", positive=True)
    ramp = _number(recipe["ramp_duration"], f"{path}.ramp_duration", nonnegative=True)
    if 2 * ramp >= duration:
        _fail(path, "has inconsistent duration and ramp_duration")
    return _CzRecipe(
        detuned_subsystem,
        duration,
        ramp,
        _number(recipe["park_detuning_ghz"], f"{path}.park_detuning_ghz"),
        _number(
            recipe["branch_tolerance_ghz"],
            f"{path}.branch_tolerance_ghz",
            nonnegative=True,
        ),
    )


def _validate_generated_provenance(value: Any, path: str) -> None:
    provenance = _mapping(value, path)
    _exact_keys(
        provenance,
        {"kind", "generator_version", "numerically_calibrated"},
        path,
    )
    if provenance["kind"] != "generated_reference_recipe":
        _fail(f"{path}.kind", "must identify a generated reference recipe")
    if (
        type(provenance["generator_version"]) is not int
        or provenance["generator_version"] != 1
    ):
        _fail(f"{path}.generator_version", "must be the integer 1")
    if provenance["numerically_calibrated"] is not False:
        _fail(f"{path}.numerically_calibrated", "must be false")


def _validate_recipes(recipes: Mapping[str, Any]) -> tuple[Any, ...]:
    _exact_keys(recipes, {"rx_ry", "iswap", "cz"}, "calibration.recipes")
    rx_ry = _validate_single_recipe(
        recipes["rx_ry"],
        "calibration.recipes.rx_ry",
        {"duration", "drag_coefficient"},
        positive_fields={"duration"},
    )
    iswap = _validate_single_recipe(
        recipes["iswap"], "calibration.recipes.iswap", {"duration"}
    )
    cz_path = "calibration.recipes.cz"
    cz = _mapping(recipes["cz"], cz_path)
    _exact_keys(cz, {"edges"}, cz_path)
    raw_edges = cz["edges"]
    if not isinstance(raw_edges, list):
        _fail(f"{cz_path}.edges", "must be an array")
    edges: dict[tuple[str, str], _CzRecipe] = {}
    for ordinal, raw in enumerate(raw_edges):
        path = f"{cz_path}.edges[{ordinal}]"
        entry = _mapping(raw, path)
        _exact_keys(entry, {"canonical_edge", "recipe"}, path)
        endpoints = entry["canonical_edge"]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            _fail(f"{path}.canonical_edge", "must name exactly two endpoints")
        key = (
            _string(endpoints[0], f"{path}.canonical_edge[0]"),
            _string(endpoints[1], f"{path}.canonical_edge[1]"),
        )
        if key[0] == key[1]:
            _fail(f"{path}.canonical_edge", "must name two distinct endpoints")
        if key[0] > key[1]:
            _fail(f"{path}.canonical_edge", "must use ascending string order")
        if key in edges:
            _fail(f"{path}.canonical_edge", "duplicates a canonical CZ edge")
        edges[key] = _validate_cz_recipe(entry["recipe"], f"{path}.recipe", key)
    return (
        rx_ry["duration"],
        rx_ry["drag_coefficient"],
        iswap["duration"],
        MappingProxyType({key: edges[key] for key in sorted(edges)}),
    )


def _parse_calibration(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "calibration"
    required = {"format", "calibration", "units", "recipes"}
    expected = required | ({"provenance"} if "provenance" in data else set())
    _exact_keys(data, expected, path)
    identity = _parse_calibration_identity(data["calibration"], f"{path}.calibration")
    if "provenance" in data:
        _validate_generated_provenance(data["provenance"], f"{path}.provenance")
    units = _mapping(data["units"], f"{path}.units")
    _exact_keys(units, set(_UNITS), f"{path}.units")
    if dict(units) != _UNITS:
        _fail(f"{path}.units", "must use the supported calibration units")
    recipes = _mapping(data["recipes"], f"{path}.recipes")
    return identity, *_validate_recipes(recipes)


_PARSERS = MappingProxyType({_FORMAT: _parse_calibration})


@dataclass(frozen=True, slots=True, init=False)
class TransmonCalibration:
    """Load transmon gate recipes from a decoded calibration document.

    All core fields are required; unknown fields are rejected. The document has:

    - ``"format"``: ``{"id": "sc.transmon_exchange_fixed_pulse",
      "version": 1}``.
    - ``"calibration"``: nonempty string ``"id"`` and ``"revision"``.
    - ``"units"``: exactly ``{"time": "ns", "frequency": "GHz",
      "dimensionless": "1"}``.
    - ``"recipes"``: ``"rx_ry"`` with positive ``"duration"`` and finite
      ``"drag_coefficient"``; ``"iswap"`` with positive ``"duration"``;
      and ``"cz"`` with an ``"edges"`` array.

    Each CZ entry names a distinct two-endpoint ``"canonical_edge"`` already
    in ascending string order. Its recipe names one absolute
    ``"detuned_subsystem"`` endpoint, a positive ``"duration"``, a
    non-negative ``"ramp_duration"`` shorter than half the duration, a finite
    ``"park_detuning_ghz"``, and a non-negative finite
    ``"branch_tolerance_ghz"``. The edge array may be empty.

    A package-generated document may also carry the strictly validated
    ``"generated_reference_recipe"`` provenance marker. This is origin
    metadata, not a runtime calibration or qualification interface.

    Args:
        document: Decoded mapping with the schema above.

    Raises:
        BackendValidationError: If the document has an unsupported format,
            missing or unknown keys, invalid units, or invalid recipe values.

    Examples:
        >>> import fatqat as fq
        >>> calibration = fq.emulator.default_transmon_calibration()
        >>> calibration.recipe_time_unit
        'ns'
    """

    _identity: _CalibrationIdentity = field(repr=False)
    _rx_ry_duration_ns: float = field(repr=False)
    _rx_ry_drag_coefficient: float = field(repr=False)
    _iswap_duration_ns: float = field(repr=False)
    _cz_by_edge: Mapping[tuple[str, str], _CzRecipe] = field(repr=False)

    __hash__ = None
    recipe_time_unit: ClassVar[str] = _UNITS["time"]
    recipe_frequency_unit: ClassVar[str] = _UNITS["frequency"]
    recipe_dimensionless_unit: ClassVar[str] = _UNITS["dimensionless"]

    def __init__(self, document: Mapping[str, Any]) -> None:
        parsed = _dispatch_document(document, "calibration", _PARSERS)
        identity, rx_duration, drag, iswap_duration, cz_by_edge = parsed
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_rx_ry_duration_ns", rx_duration)
        object.__setattr__(self, "_rx_ry_drag_coefficient", drag)
        object.__setattr__(self, "_iswap_duration_ns", iswap_duration)
        object.__setattr__(self, "_cz_by_edge", cz_by_edge)

    def _cz_recipe(self, first: str, second: str) -> _CzRecipe | None:
        key = tuple(sorted((first, second)))
        return self._cz_by_edge.get(key)


def default_transmon_calibration() -> TransmonCalibration:
    """Return the packaged reference transmon calibration."""
    document = json.loads(
        package_resources.files(__package__)
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    return TransmonCalibration(document)


__all__ = ["TransmonCalibration", "default_transmon_calibration"]
