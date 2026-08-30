"""Document-constructed superconducting transmon/exchange physics values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import pi
from types import MappingProxyType
from typing import Any, ClassVar, Self

from ...errors import BackendValidationError
from .._core.control_discovery import _ControlSelector
from .._core.document_validation import _exact_keys, _fail, _mapping, _number, _string
from .._core.model_document import (
    _FormatIdentity,
    _ModelIdentity,
    _dispatch_document,
    _parse_model_identity,
    _validate_model_document_envelope,
)
from .._core.target import _ControlAddress, _FrameAddress

_MODEL_FORMAT = _FormatIdentity("sc.transmon_exchange", 1)
_FAMILY = "sc.transmon"
_MODEL_UNITS = {"frequency": "GHz", "anharmonicity": "GHz"}


def angular_rate_from_ghz(value: float) -> float:
    """Convert an ordinary frequency in GHz to angular rate in rad/ns.

    Args:
        value: Ordinary frequency in GHz.

    Returns:
        The corresponding angular rate in radians per nanosecond.
    """
    return 2 * pi * value


@dataclass(frozen=True, slots=True)
class _Transmon:
    """Private normalized parameters of one transmon subsystem."""

    id: str
    frequency_ghz: float
    anharmonicity_ghz: float


@dataclass(frozen=True, slots=True)
class _Coupling:
    """Private normalized exchange edge between two transmons."""

    id: str
    subsystem_ids: tuple[str, str]


def _label(value: object, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendValidationError(f"{owner} must be a non-empty string")
    return value


def _drive_control(subsystem_id: str) -> _ControlAddress:
    """Select the complex-valued drive control for one subsystem.

    Args:
        subsystem_id: Nonempty subsystem label. The emulator checks that the
            label exists when the channel is used.

    Returns:
        A channel for use with ``PulseControl``.

    Raises:
        BackendValidationError: If ``subsystem_id`` is not a nonempty string.
    """
    return _ControlAddress(
        _FAMILY, "drive", (_label(subsystem_id, "control subsystem label"),)
    )


def _detuning_control(subsystem_id: str) -> _ControlAddress:
    """Select the real-valued detuning control for one subsystem.

    Args:
        subsystem_id: Nonempty subsystem label. The emulator checks that the
            label exists when the channel is used.

    Returns:
        A channel for use with ``PulseControl``.

    Raises:
        BackendValidationError: If ``subsystem_id`` is not a nonempty string.
    """
    return _ControlAddress(
        _FAMILY, "detuning", (_label(subsystem_id, "control subsystem label"),)
    )


def _exchange_control(first: str, second: str) -> _ControlAddress:
    """Select the real-valued exchange control for a subsystem pair.

    Args:
        first: Nonempty label of one endpoint.
        second: Nonempty label of the distinct other endpoint.

    Returns:
        An exchange channel for use with ``PulseControl``.

    Raises:
        BackendValidationError: If an endpoint is empty or the two endpoints
            are equal. The emulator checks for a declared coupling when the
            channel is used.
    """
    endpoints = (
        _label(first, "control subsystem label"),
        _label(second, "control subsystem label"),
    )
    if endpoints[0] == endpoints[1]:
        raise BackendValidationError(
            "exchange control requires two distinct subsystem labels"
        )
    return _ControlAddress(_FAMILY, "exchange", tuple(sorted(endpoints)))


_DRIVE_SELECTOR = _ControlSelector(
    "local", ("subsystem_id",), "complex", "rad/ns", _drive_control
)
_DETUNING_SELECTOR = _ControlSelector(
    "local", ("subsystem_id",), "real", "rad/ns", _detuning_control
)
_EXCHANGE_SELECTOR = _ControlSelector(
    "pair", ("first", "second"), "real", "rad/ns", _exchange_control
)


@dataclass(frozen=True, slots=True)
class _TransmonControls:
    """Immutable namespace for transmon control selectors."""

    drive: _ControlSelector = _DRIVE_SELECTOR
    detuning: _ControlSelector = _DETUNING_SELECTOR
    exchange: _ControlSelector = _EXCHANGE_SELECTOR


_CONTROLS = _TransmonControls()
_AVAILABLE_CONTROLS = MappingProxyType(
    {
        "drive": _CONTROLS.drive,
        "detuning": _CONTROLS.detuning,
        "exchange": _CONTROLS.exchange,
    }
)


def _parse_model(data: Mapping[str, Any]) -> tuple[Any, ...]:
    path = "physics model"
    _validate_model_document_envelope(data, path)
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
        subsystems.append(_Transmon(identifier, frequency, anharmonicity))
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
        couplings.append(_Coupling(edge_id, (first, second)))
    return identity, tuple(subsystems), tuple(couplings)


_MODEL_PARSERS = MappingProxyType({_MODEL_FORMAT: _parse_model})


@dataclass(frozen=True, slots=True, init=False, eq=False)
class TransmonModel:
    """Describe a fixed collection of three-level transmons and couplings.

    Create a model with ``from_document()``, then select drive, detuning, and
    exchange channels through ``control``.

    Examples:
        >>> import fatqat as fq
        >>> model = fq.emulator.TransmonModel.from_document(
        ...     fq.emulator.load_model_document("transmon.reference")
        ... )
        >>> model.subsystem_ids
        ('q0', 'q1')
    """

    _identity: _ModelIdentity = field(repr=False)
    _subsystems: tuple[_Transmon, ...] = field(repr=False)
    _couplings: tuple[_Coupling, ...] = field(repr=False)

    __hash__ = None
    basis_order: ClassVar[tuple[str, str, str]] = ("0", "1", "2")
    time_unit: ClassVar[str] = "ns"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Reject direct construction in favor of ``from_document()``.

        Raises:
            TypeError: Always. Physics model selection must be explicit.
        """
        raise TypeError("TransmonModel must be constructed with from_document()")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Construct a validated transmon model from a JSON document.

        Args:
            document: Mapping using the supported transmon/exchange schema.

        Returns:
            A validated transmon model.

        Raises:
            BackendValidationError: If the document has an unsupported format
                or contains invalid model data.
        """
        parsed = _dispatch_document(document, "physics model", _MODEL_PARSERS)
        identity, subsystems, couplings = parsed
        model = object.__new__(cls)
        object.__setattr__(model, "_identity", identity)
        object.__setattr__(model, "_subsystems", subsystems)
        object.__setattr__(model, "_couplings", couplings)
        return model

    @property
    def subsystem_ids(self) -> tuple[str, ...]:
        """Return subsystem labels in the model document's declared order.

        Returns:
            A tuple of labels accepted by local control selectors.
        """
        return tuple(subsystem.id for subsystem in self._subsystems)

    @property
    def control(self) -> _TransmonControls:
        """Return the ``drive``, ``detuning``, and ``exchange`` selectors.

        Returns:
            The available channel selectors.
        """
        return _CONTROLS

    @property
    def available_controls(self) -> Mapping[str, _ControlSelector]:
        """Return the channel selectors keyed by control name.

        Returns:
            A mapping containing ``drive``, ``detuning``, and ``exchange``.
        """
        return _AVAILABLE_CONTROLS

    def frame(self, subsystem_id: str) -> _FrameAddress:
        """Select the virtual drive frame for one subsystem.

        Args:
            subsystem_id: Nonempty subsystem label. The emulator checks that
                the label exists when the frame is used.

        Returns:
            A frame for ``PhaseShift`` and ``PhaseSwap``.

        Raises:
            BackendValidationError: If ``subsystem_id`` is not a nonempty
                string.
        """
        return _FrameAddress(_FAMILY, (_label(subsystem_id, "frame subsystem label"),))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransmonModel):
            return False
        return bool(
            self._identity == other._identity
            and self._subsystems == other._subsystems
            and tuple(
                (coupling.id, frozenset(coupling.subsystem_ids))
                for coupling in self._couplings
            )
            == tuple(
                (coupling.id, frozenset(coupling.subsystem_ids))
                for coupling in other._couplings
            )
        )


__all__ = [
    "TransmonModel",
    "angular_rate_from_ghz",
]
