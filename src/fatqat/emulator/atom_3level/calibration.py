"""Portable fixed-pulse calibration for the three-level atom family."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources as package_resources
from math import pi
from types import MappingProxyType
from typing import Any, ClassVar

from ...errors import BackendValidationError
from .._core.document_validation import _exact_keys, _fail, _mapping, _number
from .._core.model_document import (
    CalibrationIdentity,
    FormatIdentity,
    _dispatch_document,
    _parse_calibration_identity,
)

_FORMAT = FormatIdentity("atom.rb87_rydberg_3level_fixed_pulse", 1)
_UNITS = {
    "angular_frequency": "rad/us",
    "angle": "rad",
    "cycles": "cycle",
    "dimensionless": "1",
}


@dataclass(frozen=True, slots=True)
class _RxRyRecipe:
    omega_01: float


@dataclass(frozen=True, slots=True)
class _CzRecipe:
    omega_1r: float
    phase_amplitude: float
    phase_rate_ratio: float
    phase_offset: float
    linear_phase_rate_ratio: float
    duration_area: float
    local_z_correction: float


def _parse_calibration(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "calibration"
    _exact_keys(data, {"format", "calibration", "units", "recipes"}, path)
    identity = _parse_calibration_identity(data["calibration"], f"{path}.calibration")
    units = _mapping(data["units"], f"{path}.units")
    _exact_keys(units, set(_UNITS), f"{path}.units")
    if dict(units) != _UNITS:
        _fail(f"{path}.units", "must use the supported calibration units")
    recipes = _mapping(data["recipes"], f"{path}.recipes")
    _exact_keys(recipes, {"rx_ry", "cz"}, f"{path}.recipes")
    rx_ry = _mapping(recipes["rx_ry"], f"{path}.recipes.rx_ry")
    _exact_keys(rx_ry, {"omega_01"}, f"{path}.recipes.rx_ry")
    cz = _mapping(recipes["cz"], f"{path}.recipes.cz")
    _exact_keys(
        cz,
        {
            "omega_1r",
            "phase_amplitude",
            "phase_rate_ratio",
            "phase_offset",
            "linear_phase_rate_ratio",
            "duration_area",
            "local_z_correction",
        },
        f"{path}.recipes.cz",
    )
    return (
        identity,
        _RxRyRecipe(
            _number(rx_ry["omega_01"], f"{path}.recipes.rx_ry.omega_01", positive=True)
        ),
        _CzRecipe(
            _number(cz["omega_1r"], f"{path}.recipes.cz.omega_1r", positive=True),
            _number(cz["phase_amplitude"], f"{path}.recipes.cz.phase_amplitude"),
            _number(cz["phase_rate_ratio"], f"{path}.recipes.cz.phase_rate_ratio"),
            _number(cz["phase_offset"], f"{path}.recipes.cz.phase_offset"),
            _number(
                cz["linear_phase_rate_ratio"],
                f"{path}.recipes.cz.linear_phase_rate_ratio",
            ),
            _number(
                cz["duration_area"], f"{path}.recipes.cz.duration_area", positive=True
            ),
            _number(cz["local_z_correction"], f"{path}.recipes.cz.local_z_correction"),
        ),
    )


_PARSERS = MappingProxyType({_FORMAT: _parse_calibration})


@dataclass(frozen=True, slots=True, init=False)
class Atom3LevelCalibration:
    """Immutable portable fixed-pulse calibration recipe."""

    format: FormatIdentity = field(compare=False)
    identity: CalibrationIdentity
    _rx_ry: _RxRyRecipe = field(repr=False)
    _cz: _CzRecipe = field(repr=False)

    __hash__ = None
    angular_frequency_unit: ClassVar[str] = _UNITS["angular_frequency"]
    angle_unit: ClassVar[str] = _UNITS["angle"]
    cycles_unit: ClassVar[str] = _UNITS["cycles"]
    dimensionless_unit: ClassVar[str] = _UNITS["dimensionless"]

    def __init__(self, document: Mapping[str, Any]) -> None:
        source_format, parsed = _dispatch_document(document, "calibration", _PARSERS)
        identity, rx_ry, cz = parsed
        object.__setattr__(self, "format", source_format)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "_rx_ry", rx_ry)
        object.__setattr__(self, "_cz", cz)

    def recipe(self, name: str) -> Mapping[str, Any]:
        if name == "rx_ry":
            return MappingProxyType({"omega_01": self._rx_ry.omega_01})
        if name == "cz":
            return MappingProxyType(
                {
                    "omega_1r": self._cz.omega_1r,
                    "phase_amplitude": self._cz.phase_amplitude,
                    "phase_rate_ratio": self._cz.phase_rate_ratio,
                    "phase_offset": self._cz.phase_offset,
                    "linear_phase_rate_ratio": self._cz.linear_phase_rate_ratio,
                    "duration_area": self._cz.duration_area,
                    "local_z_correction": self._cz.local_z_correction,
                }
            )
        raise BackendValidationError(f"unknown calibration recipe {name!r}")

    @property
    def omega_01_angular_per_us(self) -> float:
        return self._rx_ry.omega_01

    @property
    def omega_1r_angular_per_us(self) -> float:
        return self._cz.omega_1r

    @property
    def phase_amplitude_rad(self) -> float:
        return self._cz.phase_amplitude

    @property
    def phase_offset_rad(self) -> float:
        return self._cz.phase_offset

    @property
    def phase_rate_ratio(self) -> float:
        return self._cz.phase_rate_ratio

    @property
    def linear_phase_rate_ratio(self) -> float:
        return self._cz.linear_phase_rate_ratio

    @property
    def duration_area_cycles(self) -> float:
        return self._cz.duration_area

    @property
    def local_z_correction_rad(self) -> float:
        return self._cz.local_z_correction

    @property
    def cz_phase_rate_angular_per_us(self) -> float:
        return self.phase_rate_ratio * self.omega_1r_angular_per_us

    @property
    def cz_linear_phase_rate_angular_per_us(self) -> float:
        return self.linear_phase_rate_ratio * self.omega_1r_angular_per_us

    @property
    def cz_duration_us(self) -> float:
        return 2 * pi * self.duration_area_cycles / self.omega_1r_angular_per_us


def default_atom_3level_calibration() -> Atom3LevelCalibration:
    """Return a fresh package-default portable three-level atom calibration."""
    document = json.loads(
        package_resources.files(__package__)
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    return Atom3LevelCalibration(document)


__all__ = ["Atom3LevelCalibration", "default_atom_3level_calibration"]
