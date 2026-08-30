"""Strict document-constructed three-level Rb87 physics model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Self

from ...errors import BackendValidationError
from .._core.control_discovery import _ControlSelector
from .._core.document_validation import _exact_keys, _fail, _mapping, _number
from .._core.model_document import (
    _FormatIdentity,
    _ModelIdentity,
    _dispatch_document,
    _parse_model_identity,
    _validate_model_document_envelope,
)
from .._core.target import _ControlAddress, _FrameAddress

_MODEL_FORMAT = _FormatIdentity("atom.rb87_rydberg_3level", 1)
_MODEL_KIND = "atom.rydberg_3level"
_FAMILY = "atom.rydberg_3level"
_BASIS = {
    "0": "5S1/2,F=1,mF=0",
    "1": "5S1/2,F=2,mF=0",
    "r": "53S1/2,mJ=+1/2",
}
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


def _raman_control(site: int) -> _ControlAddress:
    """Select the complex-valued Raman control for one site.

    Args:
        site: Nonnegative site index. The emulator checks that the site exists
            when the channel is used.

    Returns:
        A Raman channel for use with ``PulseControl``.

    Raises:
        BackendValidationError: If ``site`` is not a nonnegative built-in
            integer.
    """
    return _ControlAddress(_FAMILY, "raman_01", (_device_site(site, "atom control"),))


def _rydberg_control(site: int) -> _ControlAddress:
    """Select the complex-valued Rydberg control for one site.

    Args:
        site: Nonnegative site index. The emulator checks that the site exists
            when the channel is used.

    Returns:
        A Rydberg channel for use with ``PulseControl``.

    Raises:
        BackendValidationError: If ``site`` is not a nonnegative built-in
            integer.
    """
    return _ControlAddress(_FAMILY, "rydberg_1r", (_device_site(site, "atom control"),))


_RAMAN_SELECTOR = _ControlSelector(
    "local", ("site",), "complex", "rad/us", _raman_control
)
_RYDBERG_SELECTOR = _ControlSelector(
    "local", ("site",), "complex", "rad/us", _rydberg_control
)


@dataclass(frozen=True, slots=True)
class _Atom3Controls:
    """Immutable namespace for three-level atom control selectors."""

    raman: _ControlSelector = _RAMAN_SELECTOR
    rydberg: _ControlSelector = _RYDBERG_SELECTOR


_CONTROLS = _Atom3Controls()
_AVAILABLE_CONTROLS = MappingProxyType(
    {"raman": _CONTROLS.raman, "rydberg": _CONTROLS.rydberg}
)


def _parse_model(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "physics model"
    _validate_model_document_envelope(data, path)
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
    """Describe the three-level Rb87 physics used by the atom emulator.

    Create a model with ``from_document()``. Site coordinates are supplied
    separately through ``AtomArrangement``; select local Raman and Rydberg
    channels through ``control``.
    """

    _identity: _ModelIdentity = field(repr=False)
    species: str
    mass_u: float
    computational_states: Mapping[str, str]
    rydberg_state: str
    rydberg_coupled_state: str
    c6_angular_per_us_um6: float

    __hash__ = None
    kind: ClassVar[str] = _MODEL_KIND
    local_dimension: ClassVar[int] = 3
    mass_unit: ClassVar[str] = _MODEL_UNITS["mass"]
    distance_unit: ClassVar[str] = _MODEL_UNITS["distance"]
    time_unit: ClassVar[str] = _MODEL_UNITS["time"]
    angular_frequency_unit: ClassVar[str] = _MODEL_UNITS["angular_frequency"]
    c6_unit: ClassVar[str] = _MODEL_UNITS["c6"]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Reject direct construction in favor of ``from_document()``.

        Raises:
            TypeError: Always. Physics model selection must be explicit.
        """
        raise TypeError("Atom3LevelModel must be constructed with from_document()")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Construct a validated three-level atom model from a JSON document.

        Args:
            document: Mapping using the supported Atom3Level model schema.

        Returns:
            A validated three-level atom model.

        Raises:
            BackendValidationError: If the document has an unsupported format
                or contains invalid model data.
        """
        parsed = _dispatch_document(document, "physics model", _MODEL_PARSERS)
        identity, species, mass, c6 = parsed
        model = object.__new__(cls)
        object.__setattr__(model, "_identity", identity)
        object.__setattr__(model, "species", species)
        object.__setattr__(model, "mass_u", mass)
        object.__setattr__(
            model,
            "computational_states",
            MappingProxyType({"0": _BASIS["0"], "1": _BASIS["1"]}),
        )
        object.__setattr__(model, "rydberg_state", _BASIS["r"])
        object.__setattr__(model, "rydberg_coupled_state", "1")
        object.__setattr__(model, "c6_angular_per_us_um6", c6)
        return model

    @property
    def control(self) -> _Atom3Controls:
        """Return the local ``raman`` and ``rydberg`` selectors.

        Returns:
            The available channel selectors.
        """
        return _CONTROLS

    @property
    def available_controls(self) -> Mapping[str, _ControlSelector]:
        """Return the channel selectors keyed by control name.

        Returns:
            A mapping containing ``raman`` and ``rydberg``.
        """
        return _AVAILABLE_CONTROLS

    def frame(self, site: int) -> _FrameAddress:
        """Select the virtual frame associated with one atom site.

        Args:
            site: Nonnegative site index. The emulator checks that the site
                exists when the frame is used.

        Returns:
            A frame for ``PhaseShift`` and ``PhaseSwap``.

        Raises:
            BackendValidationError: If ``site`` is not a nonnegative built-in
                integer.
        """
        return _FrameAddress(_FAMILY, (_device_site(site, "atom frame"),))


__all__ = ["Atom3LevelModel"]
