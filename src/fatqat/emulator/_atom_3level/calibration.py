"""Fixed-pulse calibration values for the three-level atom family."""

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
    _CalibrationIdentity,
    _FormatIdentity,
    _dispatch_document,
    _parse_calibration_identity,
)

_FORMAT = _FormatIdentity("atom.rb87_rydberg_3level_fixed_pulse", 1)
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
    """Load three-level atom gate recipes from a calibration document.

    All fields are required; unknown fields are rejected. The document has:

    - ``"format"``: ``{"id": "atom.rb87_rydberg_3level_fixed_pulse",
      "version": 1}``.
    - ``"calibration"``: nonempty string ``"id"`` and ``"revision"``.
    - ``"units"``: ``"angular_frequency": "rad/us"``, ``"angle": "rad"``,
      ``"cycles": "cycle"``, and ``"dimensionless": "1"``.
    - ``"recipes"``: ``"rx_ry"`` with positive ``"omega_01"``, and ``"cz"``
      with positive ``"omega_1r"`` and ``"duration_area"`` plus finite
      ``"phase_amplitude"``, ``"phase_rate_ratio"``, ``"phase_offset"``,
      ``"linear_phase_rate_ratio"``, and ``"local_z_correction"``.

    The ``omega`` values are angular rates in rad/us. Phase amplitudes,
    offsets, and corrections are in radians; rate ratios are dimensionless;
    ``duration_area`` sets the CZ duration in cycles.

    Args:
        document: Decoded mapping with the schema above.

    Raises:
        BackendValidationError: If the document has an unsupported format,
            missing or unknown keys, invalid units, or invalid recipe values.
    """

    _identity: _CalibrationIdentity = field(repr=False)
    _rx_ry: _RxRyRecipe = field(repr=False)
    _cz: _CzRecipe = field(repr=False)

    __hash__ = None
    angular_frequency_unit: ClassVar[str] = _UNITS["angular_frequency"]
    angle_unit: ClassVar[str] = _UNITS["angle"]
    cycles_unit: ClassVar[str] = _UNITS["cycles"]
    dimensionless_unit: ClassVar[str] = _UNITS["dimensionless"]

    def __init__(self, document: Mapping[str, Any]) -> None:
        parsed = _dispatch_document(document, "calibration", _PARSERS)
        identity, rx_ry, cz = parsed
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_rx_ry", rx_ry)
        object.__setattr__(self, "_cz", cz)

    def recipe(self, name: str) -> Mapping[str, Any]:
        """Return the named ``"rx_ry"`` or ``"cz"`` recipe.

        Raises:
            BackendValidationError: If ``name`` is not ``"rx_ry"`` or
                ``"cz"``.
        """
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
        """Return the calibrated Raman angular rate in rad/us."""
        return self._rx_ry.omega_01

    @property
    def omega_1r_angular_per_us(self) -> float:
        """Return the calibrated Rydberg angular rate in rad/us."""
        return self._cz.omega_1r

    @property
    def phase_amplitude_rad(self) -> float:
        """Return the oscillating CZ phase amplitude in radians."""
        return self._cz.phase_amplitude

    @property
    def phase_offset_rad(self) -> float:
        """Return the CZ phase offset in radians."""
        return self._cz.phase_offset

    @property
    def phase_rate_ratio(self) -> float:
        """Return the oscillating CZ phase-rate ratio."""
        return self._cz.phase_rate_ratio

    @property
    def linear_phase_rate_ratio(self) -> float:
        """Return the linear CZ phase-rate ratio."""
        return self._cz.linear_phase_rate_ratio

    @property
    def duration_area_cycles(self) -> float:
        """Return the dimensionless CZ duration area in cycles."""
        return self._cz.duration_area

    @property
    def local_z_correction_rad(self) -> float:
        """Return the local post-CZ frame correction in radians."""
        return self._cz.local_z_correction

    @property
    def cz_phase_rate_angular_per_us(self) -> float:
        """Return the oscillating CZ phase rate in rad/us."""
        return self.phase_rate_ratio * self.omega_1r_angular_per_us

    @property
    def cz_linear_phase_rate_angular_per_us(self) -> float:
        """Return the linear CZ phase rate in rad/us."""
        return self.linear_phase_rate_ratio * self.omega_1r_angular_per_us

    @property
    def cz_duration_us(self) -> float:
        """Return the calibrated CZ duration in microseconds."""
        return 2 * pi * self.duration_area_cycles / self.omega_1r_angular_per_us


def default_atom_3level_calibration() -> Atom3LevelCalibration:
    """Return the packaged reference three-level atom calibration."""
    document = json.loads(
        package_resources.files(__package__)
        .joinpath("data/default_calibration.json")
        .read_text(encoding="utf-8")
    )
    return Atom3LevelCalibration(document)


__all__ = ["Atom3LevelCalibration", "default_atom_3level_calibration"]
