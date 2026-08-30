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
    detuning_operand: int
    duration_ns: float
    ramp_duration_ns: float
    detuning_ghz: float


@dataclass(frozen=True, slots=True)
class _CzOverride:
    device_operands: tuple[str, str]
    recipe: _CzRecipe


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


def _validate_cz_recipe(value: Any, path: str) -> _CzRecipe:
    recipe = _mapping(value, path)
    _exact_keys(
        recipe, {"detuning_operand", "duration", "ramp_duration", "detuning"}, path
    )
    operand = recipe["detuning_operand"]
    if type(operand) is not int or operand not in (0, 1):
        _fail(f"{path}.detuning_operand", "must be 0 or 1")
    duration = _number(recipe["duration"], f"{path}.duration", positive=True)
    ramp = _number(recipe["ramp_duration"], f"{path}.ramp_duration", nonnegative=True)
    if 2 * ramp >= duration:
        _fail(path, "has inconsistent duration and ramp_duration")
    return _CzRecipe(
        operand, duration, ramp, _number(recipe["detuning"], f"{path}.detuning")
    )


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
    cz = _mapping(recipes["cz"], "calibration.recipes.cz")
    _exact_keys(cz, {"default", "overrides"}, "calibration.recipes.cz")
    default = _validate_cz_recipe(cz["default"], "calibration.recipes.cz.default")
    raw_overrides = cz["overrides"]
    if not isinstance(raw_overrides, list):
        _fail("calibration.recipes.cz.overrides", "must be an array")
    found: set[tuple[str, str]] = set()
    overrides = []
    for ordinal, raw in enumerate(raw_overrides):
        path = f"calibration.recipes.cz.overrides[{ordinal}]"
        override = _mapping(raw, path)
        _exact_keys(override, {"device_operands", "recipe"}, path)
        endpoints = override["device_operands"]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            _fail(f"{path}.device_operands", "must name exactly two device operands")
        key = (
            _string(endpoints[0], f"{path}.device_operands[0]"),
            _string(endpoints[1], f"{path}.device_operands[1]"),
        )
        if key[0] == key[1]:
            _fail(f"{path}.device_operands", "must name two distinct device operands")
        if key in found:
            _fail(f"{path}.device_operands", "duplicates an ordered CZ override")
        found.add(key)
        overrides.append(
            _CzOverride(key, _validate_cz_recipe(override["recipe"], f"{path}.recipe"))
        )
    return (
        rx_ry["duration"],
        rx_ry["drag_coefficient"],
        iswap["duration"],
        default,
        tuple(sorted(overrides, key=lambda item: item.device_operands)),
    )


def _parse_calibration(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "calibration"
    _exact_keys(data, {"format", "calibration", "units", "recipes"}, path)
    identity = _parse_calibration_identity(data["calibration"], f"{path}.calibration")
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

    All fields are required; unknown fields are rejected. The document has:

    - ``"format"``: ``{"id": "sc.transmon_exchange_fixed_pulse",
      "version": 1}``.
    - ``"calibration"``: nonempty string ``"id"`` and ``"revision"``.
    - ``"units"``: exactly ``{"time": "ns", "frequency": "GHz",
      "dimensionless": "1"}``.
    - ``"recipes"``: ``"rx_ry"`` with positive ``"duration"`` and finite
      ``"drag_coefficient"``; ``"iswap"`` with positive ``"duration"``;
      and ``"cz"`` with one ``"default"`` recipe plus ``"overrides"``.

    A CZ recipe contains ``"detuning_operand"`` (``0`` or ``1``), positive
    ``"duration"``, non-negative ``"ramp_duration"`` shorter than half the
    duration, and finite ``"detuning"`` in GHz. Each override contains two
    distinct ordered string ``"device_operands"`` and a complete ``"recipe"``.

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
    _cz_default: _CzRecipe = field(repr=False)
    _cz_overrides: tuple[_CzOverride, ...] = field(repr=False)

    __hash__ = None
    recipe_time_unit: ClassVar[str] = _UNITS["time"]
    recipe_frequency_unit: ClassVar[str] = _UNITS["frequency"]
    recipe_dimensionless_unit: ClassVar[str] = _UNITS["dimensionless"]

    def __init__(self, document: Mapping[str, Any]) -> None:
        parsed = _dispatch_document(document, "calibration", _PARSERS)
        identity, rx_duration, drag, iswap_duration, cz_default, overrides = parsed
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_rx_ry_duration_ns", rx_duration)
        object.__setattr__(self, "_rx_ry_drag_coefficient", drag)
        object.__setattr__(self, "_iswap_duration_ns", iswap_duration)
        object.__setattr__(self, "_cz_default", cz_default)
        object.__setattr__(self, "_cz_overrides", overrides)

    def _cz_entry(self, first: str, second: str) -> _CzRecipe:
        key = (first, second)
        for override in self._cz_overrides:
            if override.device_operands == key:
                return override.recipe
        return self._cz_default

    def _cz_detuning_subsystem(self, first: str, second: str) -> str:
        return (first, second)[self._cz_entry(first, second).detuning_operand]

    def _cz_duration_ns(self, first: str, second: str) -> float:
        return self._cz_entry(first, second).duration_ns

    def _cz_ramp_duration_ns(self, first: str, second: str) -> float:
        return self._cz_entry(first, second).ramp_duration_ns

    def _cz_detuning_ghz(self, first: str, second: str) -> float:
        return self._cz_entry(first, second).detuning_ghz


def default_transmon_calibration() -> TransmonCalibration:
    """Return the packaged reference transmon calibration."""
    document = json.loads(
        package_resources.files(__package__)
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    return TransmonCalibration(document)


__all__ = ["TransmonCalibration", "default_transmon_calibration"]
