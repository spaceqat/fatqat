"""Strict document-constructed three-level Rb87 physics model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from ...errors import BackendValidationError
from .._core.document_validation import _exact_keys, _fail, _mapping, _number
from .._core.model_document import (
    FormatIdentity,
    ModelIdentity,
    _dispatch_document,
    _parse_model_identity,
)
from .._core.target import _ControlAddress, _FrameAddress

_MODEL_FORMAT = FormatIdentity("atom.rb87_rydberg_3level", 1)
_MODEL_KIND = "atom.rydberg_3level"
_FAMILY = "atom.rydberg_3level"
_BASIS = {"0": "5S1/2,F=1,mF=0", "1": "5S1/2,F=2,mF=0", "r": "53S1/2,mJ=1/2"}
_MODEL_UNITS = {
    "mass": "u",
    "distance": "um",
    "time": "us",
    "angular_frequency": "rad/us",
    "c6": "rad/us*um^6",
}


def _device_site(value: int, owner: str) -> int:
    if type(value) is not int or value < 0:
        raise BackendValidationError(f"{owner} site must be a non-negative int")
    return value


def _parse_model(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "physics model"
    _exact_keys(data, {"format", "model", "system", "units", "parameters"}, path)
    identity = _parse_model_identity(data["model"], f"{path}.model")
    system = _mapping(data["system"], f"{path}.system")
    _exact_keys(system, {"species", "basis", "transitions"}, f"{path}.system")
    if system["species"] != "Rb87":
        _fail(f"{path}.system.species", "must be 'Rb87'")
    basis = _mapping(system["basis"], f"{path}.system.basis")
    _exact_keys(basis, set(_BASIS), f"{path}.system.basis")
    if dict(basis) != _BASIS:
        _fail(f"{path}.system.basis", "must use the exact Rb87 basis identities")
    transitions = _mapping(system["transitions"], f"{path}.system.transitions")
    _exact_keys(transitions, {"rydberg"}, f"{path}.system.transitions")
    rydberg = _mapping(transitions["rydberg"], f"{path}.system.transitions.rydberg")
    _exact_keys(rydberg, {"from", "to"}, f"{path}.system.transitions.rydberg")
    if dict(rydberg) != {"from": "1", "to": "r"}:
        _fail(f"{path}.system.transitions.rydberg", "must be the transition 1 -> r")
    units = _mapping(data["units"], f"{path}.units")
    _exact_keys(units, set(_MODEL_UNITS), f"{path}.units")
    if dict(units) != _MODEL_UNITS:
        _fail(f"{path}.units", "must use the supported atom units")
    parameters = _mapping(data["parameters"], f"{path}.parameters")
    _exact_keys(parameters, {"mass", "c6"}, f"{path}.parameters")
    mass = _number(parameters["mass"], f"{path}.parameters.mass", positive=True)
    c6 = _number(parameters["c6"], f"{path}.parameters.c6")
    if c6 == 0:
        _fail(f"{path}.parameters.c6", "must be non-zero")
    return identity, "Rb87", mass, c6


_MODEL_PARSERS = MappingProxyType({_MODEL_FORMAT: _parse_model})


@dataclass(frozen=True, slots=True, init=False)
class Atom3LevelModel:
    """Immutable geometry-free three-level Rb87 blockade model."""

    format: FormatIdentity = field(compare=False)
    identity: ModelIdentity
    species: str
    mass_u: float
    computational_states: Mapping[str, str]
    rydberg_state: str
    rydberg_coupled_state: str
    c6_angular_per_us_um6: float

    __hash__ = None
    kind: ClassVar[str] = _MODEL_KIND
    local_dimension: ClassVar[int] = 3
    control_families: ClassVar[tuple[str, ...]] = (
        "raman_01",
        "rydberg_1r",
        "rydberg_blockade_interaction",
    )
    mass_unit: ClassVar[str] = _MODEL_UNITS["mass"]
    distance_unit: ClassVar[str] = _MODEL_UNITS["distance"]
    time_unit: ClassVar[str] = _MODEL_UNITS["time"]
    angular_frequency_unit: ClassVar[str] = _MODEL_UNITS["angular_frequency"]
    c6_unit: ClassVar[str] = _MODEL_UNITS["c6"]

    def __init__(self, document: Mapping[str, Any]) -> None:
        source_format, parsed = _dispatch_document(
            document, "physics model", _MODEL_PARSERS
        )
        identity, species, mass, c6 = parsed
        object.__setattr__(self, "format", source_format)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "mass_u", mass)
        object.__setattr__(
            self,
            "computational_states",
            MappingProxyType({"0": _BASIS["0"], "1": _BASIS["1"]}),
        )
        object.__setattr__(self, "rydberg_state", _BASIS["r"])
        object.__setattr__(self, "rydberg_coupled_state", "1")
        object.__setattr__(self, "c6_angular_per_us_um6", c6)

    @staticmethod
    def _control(site: int, transition: str) -> _ControlAddress:
        return _ControlAddress(
            _FAMILY, transition, (_device_site(site, "atom control"),)
        )

    def raman_control(self, site: int) -> _ControlAddress:
        return self._control(site, "raman_01")

    def rydberg_control(self, site: int) -> _ControlAddress:
        return self._control(site, "rydberg_1r")

    def frame(self, site: int) -> _FrameAddress:
        return _FrameAddress(_FAMILY, (_device_site(site, "atom frame"),))


__all__ = ["Atom3LevelModel"]
