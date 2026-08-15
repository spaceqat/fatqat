"""Document-constructed superconducting transmon/exchange physics values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import pi, sqrt
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from ...errors import BackendValidationError
from .._core.document_validation import _exact_keys, _fail, _mapping, _number, _string
from .._core.model_document import (
    FormatIdentity,
    ModelIdentity,
    _dispatch_document,
    _parse_model_identity,
)
from .._core.target import _ControlAddress, _FrameAddress
from .._core.value_validation import _freeze

_MODEL_FORMAT = FormatIdentity("sc.transmon_exchange", 1)
_MODEL_KIND = "sc.transmon"
_FAMILY = "sc.transmon"
_MODEL_UNITS = {"frequency": "GHz", "anharmonicity": "GHz"}


def angular_rate_from_ghz(value: float) -> float:
    """Convert an ordinary frequency in GHz to angular rate in rad/ns."""
    return 2 * pi * value


@dataclass(frozen=True, slots=True)
class Transmon:
    id: str
    frequency_ghz: float
    anharmonicity_ghz: float


@dataclass(frozen=True, slots=True)
class Coupling:
    id: str
    subsystem_ids: tuple[str, str]


def _label(value: object, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendValidationError(f"{owner} must be a non-empty string")
    return value


def _parse_model(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "physics model"
    _exact_keys(data, {"format", "model", "system", "units", "parameters"}, path)
    identity = _parse_model_identity(data["model"], f"{path}.model")
    system = _mapping(data["system"], f"{path}.system")
    _exact_keys(
        system, {"subsystem_type", "subsystems", "control_edges"}, f"{path}.system"
    )
    if system["subsystem_type"] != "transmon":
        _fail(f"{path}.system.subsystem_type", "must be 'transmon'")
    raw_ids = system["subsystems"]
    if not isinstance(raw_ids, list) or not raw_ids:
        _fail(f"{path}.system.subsystems", "must be a non-empty array")
    ids = tuple(
        _string(value, f"{path}.system.subsystems[{ordinal}]")
        for ordinal, value in enumerate(raw_ids)
    )
    if len(set(ids)) != len(ids):
        _fail(f"{path}.system.subsystems", "must contain unique ids")
    units = _mapping(data["units"], f"{path}.units")
    _exact_keys(units, set(_MODEL_UNITS), f"{path}.units")
    if dict(units) != _MODEL_UNITS:
        _fail(f"{path}.units", "must use GHz frequency and anharmonicity")
    parameters = _mapping(data["parameters"], f"{path}.parameters")
    _exact_keys(parameters, {"subsystems"}, f"{path}.parameters")
    values = _mapping(parameters["subsystems"], f"{path}.parameters.subsystems")
    _exact_keys(values, set(ids), f"{path}.parameters.subsystems")
    subsystems = []
    for identifier in ids:
        item_path = f"{path}.parameters.subsystems.{identifier}"
        item = _mapping(values[identifier], item_path)
        _exact_keys(item, {"frequency", "anharmonicity"}, item_path)
        frequency = _number(item["frequency"], f"{item_path}.frequency", positive=True)
        anharmonicity = _number(item["anharmonicity"], f"{item_path}.anharmonicity")
        if anharmonicity >= 0:
            _fail(f"{item_path}.anharmonicity", "must be negative")
        subsystems.append(Transmon(identifier, frequency, anharmonicity))
    edges = system["control_edges"]
    if not isinstance(edges, list):
        _fail(f"{path}.system.control_edges", "must be an array")
    edge_ids: set[str] = set()
    edge_keys: set[frozenset[str]] = set()
    couplings = []
    for ordinal, raw in enumerate(edges):
        edge_path = f"{path}.system.control_edges[{ordinal}]"
        item = _mapping(raw, edge_path)
        _exact_keys(item, {"id", "subsystems"}, edge_path)
        edge_id = _string(item["id"], f"{edge_path}.id")
        if edge_id in edge_ids:
            _fail(f"{edge_path}.id", f"duplicate control edge id {edge_id!r}")
        edge_ids.add(edge_id)
        endpoints = item["subsystems"]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            _fail(f"{edge_path}.subsystems", "must name exactly two subsystem ids")
        first = _string(endpoints[0], f"{edge_path}.subsystems[0]")
        second = _string(endpoints[1], f"{edge_path}.subsystems[1]")
        edge_key = frozenset((first, second))
        if first == second or first not in ids or second not in ids:
            _fail(
                f"{edge_path}.subsystems", "must name two distinct declared subsystems"
            )
        if edge_key in edge_keys:
            _fail(f"{edge_path}.subsystems", "duplicates an undirected control edge")
        edge_keys.add(edge_key)
        couplings.append(Coupling(edge_id, (first, second)))
    return identity, tuple(subsystems), tuple(couplings)


_MODEL_PARSERS = MappingProxyType({_MODEL_FORMAT: _parse_model})


@dataclass(frozen=True, slots=True, init=False, eq=False)
class TransmonModel:
    """Immutable engine-neutral qutrit transmon physics model."""

    format: FormatIdentity
    identity: ModelIdentity
    subsystems: tuple[Transmon, ...]
    couplings: tuple[Coupling, ...]
    annihilation: np.ndarray = field(compare=False, repr=False)
    number: np.ndarray = field(compare=False, repr=False)

    __hash__ = None
    kind: ClassVar[str] = _MODEL_KIND
    basis_order: ClassVar[tuple[str, str, str]] = ("0", "1", "2")
    local_dimension: ClassVar[int] = 3
    physical_dimension: ClassVar[int] = 3
    frequency_unit: ClassVar[str] = _MODEL_UNITS["frequency"]
    anharmonicity_unit: ClassVar[str] = _MODEL_UNITS["anharmonicity"]
    time_unit: ClassVar[str] = "ns"
    control_unit: ClassVar[str] = "rad/ns"

    def __init__(self, document: Mapping[str, Any]) -> None:
        source_format, parsed = _dispatch_document(
            document, "physics model", _MODEL_PARSERS
        )
        identity, subsystems, couplings = parsed
        annihilation = _freeze(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, sqrt(2)], [0.0, 0.0, 0.0]])
        )
        number = _freeze(np.diag([0.0, 1.0, 2.0]))
        object.__setattr__(self, "format", source_format)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "subsystems", subsystems)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "annihilation", annihilation)
        object.__setattr__(self, "number", number)

    @property
    def subsystem_ids(self) -> tuple[str, ...]:
        return tuple(subsystem.id for subsystem in self.subsystems)

    def drive_control(self, subsystem_id: str) -> _ControlAddress:
        return _ControlAddress(
            _FAMILY, "drive", (_label(subsystem_id, "control subsystem label"),)
        )

    def detuning_control(self, subsystem_id: str) -> _ControlAddress:
        return _ControlAddress(
            _FAMILY, "detuning", (_label(subsystem_id, "control subsystem label"),)
        )

    def frame(self, subsystem_id: str) -> _FrameAddress:
        return _FrameAddress(_FAMILY, (_label(subsystem_id, "frame subsystem label"),))

    def exchange_control(self, first: str, second: str) -> _ControlAddress:
        endpoints = (
            _label(first, "control subsystem label"),
            _label(second, "control subsystem label"),
        )
        if endpoints[0] == endpoints[1]:
            raise BackendValidationError(
                "exchange control requires two distinct subsystem labels"
            )
        return _ControlAddress(_FAMILY, "exchange", tuple(sorted(endpoints)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransmonModel):
            return False
        return bool(
            self.identity == other.identity
            and self.subsystems == other.subsystems
            and tuple(
                (coupling.id, frozenset(coupling.subsystem_ids))
                for coupling in self.couplings
            )
            == tuple(
                (coupling.id, frozenset(coupling.subsystem_ids))
                for coupling in other.couplings
            )
        )


__all__ = [
    "TransmonModel",
    "Transmon",
    "Coupling",
    "angular_rate_from_ghz",
]
