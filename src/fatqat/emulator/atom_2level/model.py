"""Strict, geometry-free document model for two-level atom emulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Self

from .._core.document_validation import _exact_keys, _fail, _mapping, _number, _string
from .._core.control_discovery import _ControlSelector
from .._core.model_document import (
    FormatIdentity,
    ModelIdentity,
    _dispatch_document,
    _parse_model_identity,
    _validate_model_document_envelope,
)
from .._core.target import _ControlAddress

_MODEL_FORMAT = FormatIdentity("atom.rb87_rydberg_2level", 1)
_MODEL_KIND = "atom.rydberg_2level"
_FAMILY = "atom.rydberg_2level"
_UNITS = {
    "distance": "um",
    "time": "us",
    "angular_frequency": "rad/us",
    "c6": "rad/us*um^6",
}
_LIMIT_KEYS = {
    "max_amplitude",
    "min_detuning",
    "max_detuning",
    "min_duration",
    "max_duration",
}


def _drive_control() -> _ControlAddress:
    """Select the global complex-valued Rydberg drive control.

    Returns:
        An opaque channel address for use with
        :class:`~fatqat.emulator.PulseControl`.
    """
    return _ControlAddress(_FAMILY, "drive")


def _detuning_control() -> _ControlAddress:
    """Select the global real-valued detuning control.

    Returns:
        An opaque channel address for use with
        :class:`~fatqat.emulator.PulseControl`.
    """
    return _ControlAddress(_FAMILY, "detuning")


_DRIVE_SELECTOR = _ControlSelector("global", (), "complex", "rad/us", _drive_control)
_DETUNING_SELECTOR = _ControlSelector("global", (), "real", "rad/us", _detuning_control)


@dataclass(frozen=True, slots=True)
class _Atom2Controls:
    """Immutable namespace for two-level atom control selectors."""

    drive: _ControlSelector = _DRIVE_SELECTOR
    detuning: _ControlSelector = _DETUNING_SELECTOR


_CONTROLS = _Atom2Controls()
_AVAILABLE_CONTROLS = MappingProxyType(
    {"drive": _CONTROLS.drive, "detuning": _CONTROLS.detuning}
)


@dataclass(frozen=True, slots=True)
class _GlobalControlLimits:
    max_amplitude: float | None
    min_detuning: float | None
    max_detuning: float | None
    min_duration: float | None
    max_duration: float | None


def _optional_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, positive=positive, nonnegative=nonnegative)


def _parse_limits(value: Any) -> _GlobalControlLimits:
    path = "physics model.parameters.channel_limits"
    limits = _mapping(value, path)
    _exact_keys(limits, {"rydberg_global"}, path)
    base = f"{path}.rydberg_global"
    values = _mapping(limits["rydberg_global"], base)
    _exact_keys(values, _LIMIT_KEYS, base)
    parsed = _GlobalControlLimits(
        _optional_number(
            values["max_amplitude"], f"{base}.max_amplitude", nonnegative=True
        ),
        _optional_number(values["min_detuning"], f"{base}.min_detuning"),
        _optional_number(values["max_detuning"], f"{base}.max_detuning"),
        _optional_number(values["min_duration"], f"{base}.min_duration", positive=True),
        _optional_number(values["max_duration"], f"{base}.max_duration", positive=True),
    )
    minimum_detuning = parsed.min_detuning
    maximum_detuning = parsed.max_detuning
    minimum_duration = parsed.min_duration
    maximum_duration = parsed.max_duration
    if (
        minimum_detuning is not None
        and maximum_detuning is not None
        and minimum_detuning > maximum_detuning
    ):
        _fail(base, "minimum detuning must not exceed maximum detuning")
    if (
        minimum_duration is not None
        and maximum_duration is not None
        and minimum_duration > maximum_duration
    ):
        _fail(base, "minimum duration must not exceed maximum duration")
    return parsed


def _parse_model(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "physics model"
    _validate_model_document_envelope(data, path)
    identity = _parse_model_identity(data["model"], f"{path}.model")
    system = _mapping(data["system"], f"{path}.system")
    _exact_keys(system, {"species", "basis", "transitions"}, f"{path}.system")
    if system["species"] != "Rb87":
        _fail(f"{path}.system.species", "must be 'Rb87'")
    basis = _mapping(system["basis"], f"{path}.system.basis")
    _exact_keys(basis, {"g", "r"}, f"{path}.system.basis")
    ground = _string(basis["g"], f"{path}.system.basis.g")
    rydberg = _string(basis["r"], f"{path}.system.basis.r")
    if ground == rydberg:
        _fail(f"{path}.system.basis", "ground and Rydberg states must be distinct")
    transitions = _mapping(system["transitions"], f"{path}.system.transitions")
    _exact_keys(transitions, {"rydberg"}, f"{path}.system.transitions")
    transition = _mapping(transitions["rydberg"], f"{path}.system.transitions.rydberg")
    _exact_keys(transition, {"from", "to"}, f"{path}.system.transitions.rydberg")
    if dict(transition) != {"from": "g", "to": "r"}:
        _fail(f"{path}.system.transitions.rydberg", "must be the transition g -> r")
    units = _mapping(data["units"], f"{path}.units")
    _exact_keys(units, set(_UNITS), f"{path}.units")
    if dict(units) != _UNITS:
        _fail(f"{path}.units", "must use the supported two-level atom units")
    parameters = _mapping(data["parameters"], f"{path}.parameters")
    _exact_keys(parameters, {"c6", "channel_limits"}, f"{path}.parameters")
    return (
        identity,
        "Rb87",
        ground,
        rydberg,
        _number(parameters["c6"], f"{path}.parameters.c6"),
        _parse_limits(parameters["channel_limits"]),
    )


_MODEL_PARSERS = MappingProxyType({_MODEL_FORMAT: _parse_model})


@dataclass(frozen=True, slots=True, init=False)
class Atom2LevelModel:
    """Immutable geometry-free two-level Rydberg physics model.

    Construct this value with :meth:`from_document`, then discover structural
    pulse channels through :attr:`control` or :attr:`available_controls`.
    Geometry is supplied separately to :class:`Atom2LevelEmulator`.
    """

    format: FormatIdentity = field(compare=False)
    identity: ModelIdentity
    species: str
    ground_state: str
    rydberg_state: str
    c6_angular_per_us_um6: float
    _limits: _GlobalControlLimits

    __hash__ = None
    kind: ClassVar[str] = _MODEL_KIND
    basis_order: ClassVar[tuple[str, str]] = ("g", "r")
    local_dimension: ClassVar[int] = 2
    interaction_law: ClassVar[str] = "C6/R^6"
    distance_unit: ClassVar[str] = _UNITS["distance"]
    time_unit: ClassVar[str] = _UNITS["time"]
    angular_frequency_unit: ClassVar[str] = _UNITS["angular_frequency"]
    c6_unit: ClassVar[str] = _UNITS["c6"]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Reject direct construction in favor of :meth:`from_document`.

        Raises:
            TypeError: Always. Physics model selection must be explicit.
        """
        raise TypeError("Atom2LevelModel must be constructed with from_document()")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Construct a validated two-level atom model from a JSON document.

        Args:
            document: Mapping using the supported Atom2Level model schema.

        Returns:
            An immutable, geometry-free physics model.

        Raises:
            BackendValidationError: If the document has an unsupported format
                or contains invalid model data.
        """
        source_format, parsed = _dispatch_document(
            document, "physics model", _MODEL_PARSERS
        )
        (
            identity,
            species,
            ground,
            rydberg,
            c6,
            limits,
        ) = parsed
        model = object.__new__(cls)
        object.__setattr__(model, "format", source_format)
        object.__setattr__(model, "identity", identity)
        object.__setattr__(model, "species", species)
        object.__setattr__(model, "ground_state", ground)
        object.__setattr__(model, "rydberg_state", rydberg)
        object.__setattr__(model, "c6_angular_per_us_um6", c6)
        object.__setattr__(model, "_limits", limits)
        return model

    @property
    def control(self) -> _Atom2Controls:
        """Return the immutable namespace of supported control selectors.

        Returns:
            A family-owned namespace containing ``drive`` and ``detuning``.
        """
        return _CONTROLS

    @property
    def available_controls(self) -> Mapping[str, _ControlSelector]:
        """Return inspectable selectors keyed by their public control names.

        Returns:
            An immutable mapping whose values are the selectors on
            :attr:`control`.
        """
        return _AVAILABLE_CONTROLS


__all__ = ["Atom2LevelModel"]
