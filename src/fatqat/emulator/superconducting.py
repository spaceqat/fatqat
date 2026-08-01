"""Data-only superconducting transmon/exchange model foundations.

This module deliberately contains no solver, global tensor operator, or QuTiP
object.  It validates a durable model snapshot, builds immutable local-qutrit
facts and opaque model handles, and loads a separately identity-bound
calibration profile for the later pulse-realization layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, pi, sqrt
from types import MappingProxyType
from typing import Any, ClassVar, cast

import numpy as np

from ..errors import BackendValidationError
from ._validation import _freeze
from .model_contract import ControlChannel, Frame, ResourceClaim

_MODEL_FORMAT = "fatqat.physics-model"
_CALIBRATION_FORMAT = "fatqat.calibration"
_SC_BUILDER_ID = "sc.transmon_exchange"
_SC_BUILDER_VERSION = 1
_SCHEMA_VERSION = 1


def _fail(path: str, message: str) -> None:
    """Raise one path-qualified model/calibration validation failure."""
    raise BackendValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a mapping with string keys and return it unchanged."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(path, "must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    """Require an exact object schema, reporting missing and unknown keys."""
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)!r}")
        if extra:
            detail.append(f"unknown {sorted(extra)!r}")
        _fail(path, "; ".join(detail))


def _data_only(value: Any, path: str) -> None:
    """Reject persistence values outside the JSON data model."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _data_only(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _data_only(child, f"{path}.{key}")
        return
    _fail(path, f"must contain JSON data only, not {type(value).__name__}")


def _string(value: Any, path: str) -> str:
    """Require a non-empty string persistence value."""
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _version(value: Any, path: str) -> int:
    """Require a positive integer schema/builder version."""
    if type(value) is not int or value < 1:
        _fail(path, "must be a positive integer")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    """Normalize one finite numeric value, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not isfinite(result):
        _fail(path, "must be finite")
    if positive and result <= 0:
        _fail(path, "must be positive")
    return result


def angular_rate_from_ghz(value: float) -> float:
    """Convert a stored GHz model/calibration parameter to an angular rate.

    Persisted model and calibration numbers are ordinary frequencies in GHz,
    while every Hamiltonian coefficient the solver consumes is an *angular*
    rate in ``rad/ns`` (see :attr:`SCTransmonModel.control_unit`). Since
    ``1 GHz == 1/ns``, the bridge is exactly a factor of ``2*pi``.

    This is the one definition of that convention. Realization rules and the
    solver adapter both call it rather than writing ``2 * pi * x_ghz`` inline,
    so the factor cannot drift between the controls a rule emits and the drift
    term the adapter builds.
    """
    return 2 * pi * value


def _freeze_data(value: Any) -> Any:
    """Recursively freeze JSON mappings/lists while preserving scalar values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_data(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_data(child) for child in value)
    return value


@dataclass(frozen=True)
class BuilderIdentity:
    """Trusted builder key persisted in model and calibration documents.

    The pair ``(id, version)`` selects package-owned executable builder code;
    persisted documents contain data only and cannot provide a callback.
    """

    id: str
    version: int

    @classmethod
    def from_mapping(cls, value: Any, path: str = "builder") -> BuilderIdentity:
        """Validate and parse one exact ``{"id", "version"}`` object."""
        data = _mapping(value, path)
        _exact_keys(data, {"id", "version"}, path)
        return cls(
            id=_string(data["id"], f"{path}.id"),
            version=_version(data["version"], f"{path}.version"),
        )


@dataclass(frozen=True)
class ModelIdentity:
    """Stable identity for one immutable persisted model snapshot.

    ``id`` names the modeled device and ``revision`` distinguishes immutable
    snapshots of that device. Calibration documents must repeat both values.
    """

    id: str
    revision: str

    @classmethod
    def from_mapping(cls, value: Any, path: str = "model") -> ModelIdentity:
        """Validate and parse one exact ``{"id", "revision"}`` object."""
        data = _mapping(value, path)
        _exact_keys(data, {"id", "revision"}, path)
        return cls(
            id=_string(data["id"], f"{path}.id"),
            revision=_string(data["revision"], f"{path}.revision"),
        )


@dataclass(frozen=True)
class ModelKey:
    """Complete builder-and-snapshot identity.

    The key binds calibration data and every opaque resource/control/frame
    handle to the model from which it was created.
    """

    builder: BuilderIdentity
    model: ModelIdentity


@dataclass(frozen=True)
class PhysicsModelSpec:
    """Validated, data-only physics-model persistence envelope.

    This is an internal intermediate used by :func:`load_physics_model` to
    dispatch a versioned document to package-trusted builder code. Application
    code normally passes the original mapping directly to that loader.
    """

    builder: BuilderIdentity
    model: ModelIdentity
    parameters: Mapping[str, Any]
    format: ClassVar[str] = _MODEL_FORMAT
    schema_version: ClassVar[int] = _SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, document: Any) -> PhysicsModelSpec:
        """Parse a JSON-compatible model document with an exact schema.

        Unknown/missing keys, unsupported format or schema versions, and
        non-data values are rejected before any builder is invoked.
        """
        _data_only(document, "physics model")
        data = _mapping(document, "physics model")
        _exact_keys(
            data,
            {"format", "schema_version", "builder", "model", "parameters"},
            "physics model",
        )
        if data["format"] != _MODEL_FORMAT:
            _fail("physics model.format", f"unsupported format {data['format']!r}")
        if data["schema_version"] != _SCHEMA_VERSION:
            _fail(
                "physics model.schema_version",
                f"unsupported schema version {data['schema_version']!r}",
            )
        return cls(
            builder=BuilderIdentity.from_mapping(data["builder"]),
            model=ModelIdentity.from_mapping(data["model"]),
            parameters=MappingProxyType(
                dict(_mapping(data["parameters"], "physics model.parameters"))
            ),
        )

    @property
    def key(self) -> ModelKey:
        """Return the combined builder and model identity."""
        return ModelKey(self.builder, self.model)


@dataclass(frozen=True)
class Transmon:
    """One model-local fixed-qutrit transmon's durable numerical facts.

    ``frequency_ghz`` is the nominal 0-to-1 transition frequency and defines
    this subsystem's implicit resonant rotating-frame carrier. The current
    solver uses ``Delta_i = 0``, so changing this value alone does not
    numerically change simulated dynamics. A future frame-explicit model may
    consume it as a Hamiltonian or control parameter under a new model version.
    """

    id: str
    frequency_ghz: float
    anharmonicity_ghz: float


@dataclass(frozen=True)
class Coupling:
    """One undirected model edge supporting controlled exchange operations.

    ``subsystem_ids`` records topology, not a continuously active Hamiltonian
    term. Exchange is present only when a realization drives the edge's
    exchange control channel.
    """

    id: str
    subsystem_ids: tuple[str, str]


@dataclass(frozen=True)
class SubsystemResourceRef(ResourceClaim):
    """Opaque physical subsystem resource minted by one :class:`SCTransmonModel`."""

    model_key: ModelKey
    ordinal: int
    # Included in equality/hash on purpose: two refs with the same public
    # model_key/ordinal but minted by different SCTransmonModel instances (e.g.
    # two builds from the same persisted key) must stay distinguishable, so a
    # foreign handle is unequal to a same-key native one rather than merely
    # rejected at bind time.
    _token: object = field(repr=False)


@dataclass(frozen=True)
class ControlChannelRef(ControlChannel):
    """Opaque physical control channel minted by one model.

    ``kind`` is ``"drive"``, ``"detuning"``, or ``"exchange"``. Construct
    controls from a model accessor instead of instantiating this class: the
    model also verifies object identity when a pulse is bound.
    """

    model_key: ModelKey
    ordinal: int
    kind: str
    _token: object = field(repr=False)


@dataclass(frozen=True)
class FrameRef(Frame):
    """Opaque virtual-frame handle minted by one model.

    Obtain it from :meth:`SCTransmonModel.frame`; direct construction produces a
    foreign handle that the model rejects.
    """

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False)


@dataclass(frozen=True)
class CouplingRef(ResourceClaim):
    """Opaque pair-resource handle minted by one model.

    Used solely for scheduling-conflict resource claims on a declared
    coupling edge; it is never a sampled child's channel. See
    :class:`ControlChannelRef` (``kind="exchange"``) for the physical
    exchange-drive channel.
    """

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False)


@dataclass(frozen=True)
class SCTransmonModel:
    """Immutable, engine-neutral superconducting transmon model.

    Instances are returned by :func:`load_physics_model`; applications should
    not construct them directly. The model contains ordered local qutrit facts,
    declared exchange topology, read-only local operators, and opaque handles
    used by custom pulse implementation rules. It contains no QuTiP objects,
    tensor-expanded operators, calibration values, or solver state.

    Program qubits bind to ``subsystem_ids`` in declaration order. Opaque
    handles are deliberately model-instance-specific: even a separately
    loaded model with the same persisted identity cannot bind another
    instance's handles.

    This model satisfies the private
    :class:`~fatqat.emulator.model_contract.PhysicsModel` protocol structurally;
    it does not inherit from it. Its handle types do inherit the abstract
    marker kinds so the model-neutral pulse layers can discriminate a control
    channel from a resource claim without importing this module.

    Attributes:
        key: Complete builder and snapshot identity.
        subsystems: Ordered :class:`Transmon` records.
        couplings: Declared undirected :class:`Coupling` edges.
        annihilation: Read-only local qutrit lowering matrix. The raising
            operator is its conjugate transpose and is not stored separately.
        number: Read-only local qutrit number matrix.
    """

    key: ModelKey
    subsystems: tuple[Transmon, ...]
    couplings: tuple[Coupling, ...]
    annihilation: np.ndarray
    number: np.ndarray
    _token: object = field(repr=False, compare=False)
    _resources: tuple[SubsystemResourceRef, ...] = field(repr=False)
    _drive_controls: tuple[ControlChannelRef, ...] = field(repr=False)
    _detuning_controls: tuple[ControlChannelRef, ...] = field(repr=False)
    _exchange_controls: tuple[ControlChannelRef, ...] = field(repr=False)
    _frames: tuple[FrameRef, ...] = field(repr=False)
    _coupling_refs: tuple[CouplingRef, ...] = field(repr=False)

    @property
    def time_unit(self) -> str:
        """Return the model's pulse-time coordinate (``"ns"``).

        Applies to ``duration``, every control's ``tlist``, and
        ``start_offset``.
        """
        return "ns"

    @property
    def control_unit(self) -> str:
        """Return the unit of every sampled control coefficient (``"rad/ns"``).

        An *angular* rate, not an ordinary frequency: a rule that emits GHz
        is wrong by a factor of ``2*pi``, and nothing can detect that from the
        samples alone. Stored GHz parameters must be converted with
        :func:`angular_rate_from_ghz` before they become coefficients.

        The unit is the same for all three channel kinds. A drive channel's
        complex samples carry the two quadratures; both components are in this
        unit.
        """
        return "rad/ns"

    @property
    def subsystem_ids(self) -> tuple[str, ...]:
        """Return subsystem identifiers in model/binding order."""
        return tuple(subsystem.id for subsystem in self.subsystems)

    @property
    def physical_dimension(self) -> int:
        """Return the local physical dimension, fixed at three."""
        return 3

    def resource(self, subsystem_id: str) -> SubsystemResourceRef:
        """Return the scheduling/resource handle for one subsystem.

        Raises:
            BackendValidationError: If ``subsystem_id`` is not declared.
        """
        return self._resources[self._subsystem_ordinal(subsystem_id)]

    def drive_control(self, subsystem_id: str) -> ControlChannelRef:
        """Return the complex local drive channel for one transmon."""
        return self._drive_controls[self._subsystem_ordinal(subsystem_id)]

    def detuning_control(self, subsystem_id: str) -> ControlChannelRef:
        """Return the real local-frequency-shift channel for one transmon."""
        return self._detuning_controls[self._subsystem_ordinal(subsystem_id)]

    def frame(self, subsystem_id: str) -> FrameRef:
        """Return the virtual-drive frame handle for one transmon."""
        return self._frames[self._subsystem_ordinal(subsystem_id)]

    def coupling(self, first: str, second: str) -> CouplingRef:
        """Return the pair-resource handle for one declared coupling edge.

        This is a scheduling-conflict resource claim only; it is never a
        sampled child's channel. Use :meth:`exchange_control` for the
        physical exchange-drive channel on the same edge.
        """
        return self._coupling_refs[self._coupling_ordinal(first, second)]

    def exchange_control(self, first: str, second: str) -> ControlChannelRef:
        """Return the physical exchange-drive channel for one coupling edge."""
        return self._exchange_controls[self._coupling_ordinal(first, second)]

    def _coupling_ordinal(self, first: str, second: str) -> int:
        """Resolve an undirected subsystem pair to its coupling ordinal."""
        edge = frozenset((first, second))
        for ordinal, coupling in enumerate(self.couplings):
            if frozenset(coupling.subsystem_ids) == edge:
                return ordinal
        raise BackendValidationError(
            f"model has no declared coupling edge {first!r}-{second!r}"
        )

    def bind_resource(self, reference: ResourceClaim) -> int:
        """Validate a resource handle and return its model ordinal."""
        return self._bind(reference, SubsystemResourceRef, self._resources, "resource")

    def bind_control(self, reference: ControlChannel) -> int:
        """Validate a control handle and return its kind-local model ordinal.

        Drive and detuning ordinals index ``subsystems``; exchange ordinals
        index ``couplings``.
        """
        kind = reference.kind if isinstance(reference, ControlChannelRef) else None
        controls = {
            "drive": self._drive_controls,
            "detuning": self._detuning_controls,
            "exchange": self._exchange_controls,
        }.get(kind, ())
        return self._bind(reference, ControlChannelRef, controls, "control")

    def bind_frame(self, reference: Frame) -> int:
        """Validate a frame handle and return its subsystem ordinal."""
        return self._bind(reference, FrameRef, self._frames, "frame")

    def bind_coupling(self, reference: ResourceClaim) -> int:
        """Validate a coupling-resource handle and return its edge ordinal."""
        return self._bind(reference, CouplingRef, self._coupling_refs, "coupling")

    def bind_claim(self, reference: ResourceClaim) -> int:
        """Validate any resource-claim handle and return its model ordinal.

        Dispatches between this model's two claim kinds so a `PulseBlock` can
        verify declared claims without knowing that a transmon model has both
        per-subsystem and per-edge resources.
        """
        if isinstance(reference, CouplingRef):
            return self.bind_coupling(reference)
        return self.bind_resource(reference)

    def _bound_control(self, channel: ControlChannel) -> tuple[ControlChannelRef, int]:
        """Bind ``channel`` and return it narrowed, with its model ordinal.

        The protocol hands these methods an abstract `ControlChannel`, but the
        transmon-specific logic below reads ``kind``. `bind_control` has
        already rejected anything that is not one of this model's own
        `ControlChannelRef`s, so the cast restates a checked fact rather than
        assuming one.
        """
        ordinal = self.bind_control(channel)
        return cast(ControlChannelRef, channel), ordinal

    def required_claims_for_control(
        self, channel: ControlChannel
    ) -> frozenset[ResourceClaim]:
        """Return every resource a pulse driving ``channel`` must claim.

        Driving one transmon's drive or detuning channel implicates only that
        subsystem. Driving an edge's exchange channel implicates both
        endpoints *and* the pair resource, because a concurrent block on
        either endpoint would corrupt the interaction. Keeping this topology
        here is what lets :class:`~fatqat.emulator.pulse.PulseBlock` check
        claim coverage without knowing what an exchange channel is.
        """
        control, ordinal = self._bound_control(channel)
        if control.kind == "exchange":
            coupling = self.couplings[ordinal]
            return frozenset(
                {self.resource(subsystem_id) for subsystem_id in coupling.subsystem_ids}
                | {self.coupling(*coupling.subsystem_ids)}
            )
        return frozenset({self.resource(self.subsystem_ids[ordinal])})

    def validate_control_coefficients(
        self, channel: ControlChannel, coefficients: np.ndarray
    ) -> None:
        """Reject an envelope this transmon channel cannot physically realize.

        A drive channel's complex samples encode its two quadratures, so any
        complex envelope is legal. Detuning and exchange channels multiply a
        Hermitian generator directly and therefore require real samples;
        rejecting that here, when a block binds to the model, means ``run()``
        and ``propagator()`` report the identical error at lowering instead of
        one of them failing later inside solver binding.
        """
        control, _ordinal = self._bound_control(channel)
        if control.kind == "drive":
            return
        if not np.allclose(np.asarray(coefficients).imag, 0.0, atol=1e-12, rtol=0.0):
            raise BackendValidationError(
                f"{control.kind} pulse coefficients must be real"
            )

    def _subsystem_ordinal(self, subsystem_id: str) -> int:
        """Resolve a declared subsystem ID to its ordered model ordinal."""
        try:
            return self.subsystem_ids.index(subsystem_id)
        except ValueError:
            raise BackendValidationError(
                f"unknown model subsystem {subsystem_id!r}"
            ) from None

    def _bind(
        self, reference: Any, kind: type, refs: tuple[Any, ...], name: str
    ) -> int:
        """Validate exact handle provenance, kind, ordinal, and object identity."""
        if (
            not isinstance(reference, kind)
            or reference.model_key != self.key
            or reference._token is not self._token
            or not 0 <= reference.ordinal < len(refs)
            or refs[reference.ordinal] is not reference
        ):
            raise BackendValidationError(f"unknown or foreign {name} reference")
        return reference.ordinal


class PhysicsModelBuilderRegistry:
    """Internal registry containing only package-trusted model builders.

    Persistence documents select a builder by identity but never register or
    deserialize executable code. The package-level registry is populated at
    import time and consumed by :func:`load_physics_model`.
    """

    def __init__(self) -> None:
        self._builders: dict[BuilderIdentity, SCTransmonExchangeBuilder] = {}

    def register(self, builder: SCTransmonExchangeBuilder) -> None:
        """Register one trusted builder under its unique identity."""
        identity = builder.identity
        if identity in self._builders:
            raise ValueError(
                f"builder already registered: {identity.id}/{identity.version}"
            )
        self._builders[identity] = builder

    def resolve(self, identity: BuilderIdentity) -> SCTransmonExchangeBuilder:
        """Return the trusted builder selected by a persisted identity."""
        try:
            return self._builders[identity]
        except KeyError:
            raise BackendValidationError(
                f"unsupported physics-model builder {identity.id!r} version {identity.version}"
            ) from None

    def build(self, spec: PhysicsModelSpec) -> SCTransmonModel:
        """Resolve ``spec.builder`` and build the immutable model."""
        return self.resolve(spec.builder).build(spec)


class SCTransmonExchangeBuilder:
    """Trusted builder for fixed-qutrit, arbitrary-graph transmon models.

    Normal applications call :func:`load_physics_model`, which selects this
    builder from the document's identity. The builder validates GHz subsystem
    units, negative anharmonicities, unique IDs, and unique undirected coupling
    edges, then mints model-instance-specific resource handles.

    Coupling records declare topology for controlled exchange pulses; they do
    not add residual always-on exchange to the model Hamiltonian.
    """

    identity = BuilderIdentity(_SC_BUILDER_ID, _SC_BUILDER_VERSION)

    def build(self, spec: PhysicsModelSpec) -> SCTransmonModel:
        """Build an immutable local-qutrit model from a validated spec.

        Args:
            spec: Parsed spec selecting this builder identity.

        Returns:
            A fresh :class:`SCTransmonModel` with read-only local operators and
            newly minted opaque handles.

        Raises:
            BackendValidationError: If the builder identity, units, subsystem
                data, or coupling topology are invalid.
        """
        if spec.builder != self.identity:
            raise BackendValidationError(
                "SC builder received a foreign model specification"
            )
        parameters = _mapping(spec.parameters, "physics model.parameters")
        _exact_keys(
            parameters, {"units", "subsystems", "couplings"}, "physics model.parameters"
        )
        self._validate_units(parameters["units"])
        subsystems = self._build_subsystems(parameters["subsystems"])
        couplings = self._build_couplings(parameters["couplings"], subsystems)
        annihilation = _freeze(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, sqrt(2)], [0.0, 0.0, 0.0]])
        )
        number = _freeze(np.diag([0.0, 1.0, 2.0]))
        token = object()
        resources = tuple(
            SubsystemResourceRef(spec.key, ordinal, token)
            for ordinal in range(len(subsystems))
        )
        controls = tuple(
            ControlChannelRef(spec.key, ordinal, "drive", token)
            for ordinal in range(len(subsystems))
        )
        detuning_controls = tuple(
            ControlChannelRef(spec.key, ordinal, "detuning", token)
            for ordinal in range(len(subsystems))
        )
        # Coupling-sized, not subsystem-sized: one exchange-drive channel per
        # declared edge, ordinal-aligned with `couplings`/`_coupling_refs`.
        exchange_controls = tuple(
            ControlChannelRef(spec.key, ordinal, "exchange", token)
            for ordinal in range(len(couplings))
        )
        frames = tuple(
            FrameRef(spec.key, ordinal, token) for ordinal in range(len(subsystems))
        )
        coupling_refs = tuple(
            CouplingRef(spec.key, ordinal, token) for ordinal in range(len(couplings))
        )
        return SCTransmonModel(
            key=spec.key,
            subsystems=subsystems,
            couplings=couplings,
            annihilation=annihilation,
            number=number,
            _token=token,
            _resources=resources,
            _drive_controls=controls,
            _detuning_controls=detuning_controls,
            _exchange_controls=exchange_controls,
            _frames=frames,
            _coupling_refs=coupling_refs,
        )

    @staticmethod
    def _validate_units(value: Any) -> None:
        """Require the v1 transmon frequency/anharmonicity GHz unit schema."""
        units = _mapping(value, "physics model.parameters.units")
        _exact_keys(units, {"subsystems"}, "physics model.parameters.units")
        subsystem_units = _mapping(
            units["subsystems"], "physics model.parameters.units.subsystems"
        )
        _exact_keys(
            subsystem_units,
            {"frequency", "anharmonicity"},
            "physics model.parameters.units.subsystems",
        )
        if (
            subsystem_units["frequency"] != "GHz"
            or subsystem_units["anharmonicity"] != "GHz"
        ):
            _fail(
                "physics model.parameters.units.subsystems",
                "frequency and anharmonicity must use GHz",
            )

    @staticmethod
    def _build_subsystems(value: Any) -> tuple[Transmon, ...]:
        """Validate and build ordered, uniquely named transmon records."""
        if not isinstance(value, list) or not value:
            _fail("physics model.parameters.subsystems", "must be a non-empty array")
        subsystems = []
        ids: set[str] = set()
        for ordinal, raw in enumerate(value):
            path = f"physics model.parameters.subsystems[{ordinal}]"
            item = _mapping(raw, path)
            _exact_keys(item, {"id", "frequency", "anharmonicity"}, path)
            identifier = _string(item["id"], f"{path}.id")
            if identifier in ids:
                _fail(f"{path}.id", f"duplicate subsystem id {identifier!r}")
            ids.add(identifier)
            frequency = _number(item["frequency"], f"{path}.frequency", positive=True)
            anharmonicity = _number(item["anharmonicity"], f"{path}.anharmonicity")
            if anharmonicity >= 0:
                _fail(f"{path}.anharmonicity", "must be negative")
            subsystems.append(Transmon(identifier, frequency, anharmonicity))
        return tuple(subsystems)

    @staticmethod
    def _build_couplings(
        value: Any, subsystems: tuple[Transmon, ...]
    ) -> tuple[Coupling, ...]:
        """Validate and build unique undirected edges over known subsystems."""
        if not isinstance(value, list):
            _fail("physics model.parameters.couplings", "must be an array")
        known = {subsystem.id for subsystem in subsystems}
        ids: set[str] = set()
        edges: set[frozenset[str]] = set()
        couplings = []
        for ordinal, raw in enumerate(value):
            path = f"physics model.parameters.couplings[{ordinal}]"
            item = _mapping(raw, path)
            _exact_keys(item, {"id", "subsystems"}, path)
            identifier = _string(item["id"], f"{path}.id")
            if identifier in ids:
                _fail(f"{path}.id", f"duplicate coupling id {identifier!r}")
            ids.add(identifier)
            endpoints = item["subsystems"]
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                _fail(
                    f"{path}.subsystems",
                    "must be an array of exactly two subsystem ids",
                )
            first = _string(endpoints[0], f"{path}.subsystems[0]")
            second = _string(endpoints[1], f"{path}.subsystems[1]")
            if first == second or first not in known or second not in known:
                _fail(
                    f"{path}.subsystems", "must name two distinct declared subsystems"
                )
            edge = frozenset((first, second))
            if edge in edges:
                _fail(f"{path}.subsystems", "duplicates an undirected coupling edge")
            edges.add(edge)
            couplings.append(Coupling(identifier, (first, second)))
        return tuple(couplings)


PHYSICS_MODEL_BUILDERS = PhysicsModelBuilderRegistry()
PHYSICS_MODEL_BUILDERS.register(SCTransmonExchangeBuilder())


def load_physics_model(document: Any) -> SCTransmonModel:
    """Parse and build one trusted superconducting physics-model document.

    ``document`` must contain JSON-compatible data using the versioned
    ``fatqat.physics-model`` envelope. Its builder identity selects only
    package-registered code; callbacks and arbitrary Python objects are
    rejected. The returned model is immutable and may be shared across
    backend instances.

    Args:
        document: Mapping decoded from a model JSON document.

    Returns:
        A new immutable :class:`SCTransmonModel`.

    Raises:
        BackendValidationError: If the envelope, builder, units, subsystem
            parameters, or coupling topology are invalid.
    """
    return PHYSICS_MODEL_BUILDERS.build(PhysicsModelSpec.from_mapping(document))


@dataclass(frozen=True)
class CalibrationSpec:
    """Immutable calibration recipe data bound to one model snapshot.

    Instances are returned by :func:`load_calibration_spec`; applications
    should not construct them directly. Recipe mappings are recursively frozen
    and contain data only. The built-in realization reads ``rx_ry``, ``iswap``,
    and per-edge ``cz`` recipes; virtual ``RZ`` has no calibration recipe.

    Attributes:
        key: Exact model identity copied from the calibration document.
        recipes: Recursively immutable recipe mapping.
    """

    key: ModelKey
    recipes: Mapping[str, Any]
    format: ClassVar[str] = _CALIBRATION_FORMAT
    schema_version: ClassVar[int] = _SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, document: Any, model: SCTransmonModel) -> CalibrationSpec:
        """Validate and bind a calibration document to ``model``."""
        _data_only(document, "calibration")
        data = _mapping(document, "calibration")
        _exact_keys(
            data,
            {"format", "schema_version", "builder", "model", "recipes"},
            "calibration",
        )
        if data["format"] != _CALIBRATION_FORMAT:
            _fail("calibration.format", f"unsupported format {data['format']!r}")
        if data["schema_version"] != _SCHEMA_VERSION:
            _fail(
                "calibration.schema_version",
                f"unsupported schema version {data['schema_version']!r}",
            )
        key = ModelKey(
            BuilderIdentity.from_mapping(data["builder"]),
            ModelIdentity.from_mapping(data["model"]),
        )
        if key != model.key:
            raise BackendValidationError(
                "calibration identity does not match the physics model"
            )
        recipes = _mapping(data["recipes"], "calibration.recipes")
        cls._validate_recipes(recipes, model)
        return cls(key=key, recipes=_freeze_data(dict(recipes)))

    def recipe(self, name: str) -> Mapping[str, Any]:
        """Return one immutable named recipe.

        Raises:
            BackendValidationError: If ``name`` is not present.
        """
        try:
            return self.recipes[name]
        except KeyError:
            raise BackendValidationError(
                f"unknown calibration recipe {name!r}"
            ) from None

    @staticmethod
    def _validate_recipes(recipes: Mapping[str, Any], model: SCTransmonModel) -> None:
        """Validate all built-in recipe families and exact CZ edge coverage."""
        # TODO(calibration-schema-ownership): this method hard-codes the recipe
        # schema of one specific realization - the built-in rules in
        # `superconducting_realization.py` - while `PulseImplementationMap` is
        # the documented way to replace a realization. The `_exact_keys` call
        # below therefore rejects any recipe a *custom* rule would need, so a
        # user-authored CZ cannot carry calibrated parameters in the
        # calibration document at all. Loosening the check is not the answer:
        # strict, exact-schema validation is deliberate (see
        # `test_calibration_rejects_an_rz_recipe_including_an_arbitrary_scale`
        # and `..._cz_phase_corrections_as_an_unknown_field`).
        #
        # The blocker is lifecycle, not validation. A calibration is loaded by
        # `load_calibration_spec(document, model)` *before* any
        # `PulseImplementationMap` is chosen - the map is not selected until
        # `Emulator.__init__` - so at load time there is nothing to ask
        # which recipes are legal. Two coherent resolutions, differing in when
        # an invalid recipe is reported:
        #
        #   1. Keep envelope/identity validation here and move realization
        #      schema validation to `Emulator.__init__`, where the map is
        #      known. An invalid recipe then surfaces at backend construction,
        #      not at load.
        #   2. Give `load_calibration_spec` an explicit schema/validator
        #      argument supplied by the realization (or its map/factory).
        #      Load-time reporting is preserved, at the cost of a wider public
        #      loader signature and a caller that must pass the two together.
        #
        # That user-visible difference should be settled as a design decision
        # (an ADR alongside 0021, which assigned calibration ownership) before
        # either is implemented. Do not resolve it by relaxing `_exact_keys`.
        #
        # RZ has no recipe: it realizes as an exact virtual frame rotation
        # (see superconducting_realization._rz_definition), not a calibrated
        # physical gate, so it carries no calibration degree of freedom to
        # validate.
        _exact_keys(recipes, {"rx_ry", "iswap", "cz"}, "calibration.recipes")
        # No stored amplitude: for a requested angle theta and this duration T,
        # the Hann/DRAG peak is exactly theta/T (architecture doc Sec. 5.1), so
        # realization derives it rather than reading a calibrated constant.
        CalibrationSpec._validate_single_qubit(
            recipes["rx_ry"],
            "calibration.recipes.rx_ry",
            {"duration_ns", "drag_coefficient"},
            positive_fields={"duration_ns"},
        )
        # iSWAP is a fixed-angle gate (area = pi/2), so its exchange peak is
        # likewise fully determined by duration_ns, not stored calibration data.
        CalibrationSpec._validate_single_qubit(
            recipes["iswap"],
            "calibration.recipes.iswap",
            {"duration_ns"},
        )
        cz = _mapping(recipes["cz"], "calibration.recipes.cz")
        _exact_keys(cz, {"edges"}, "calibration.recipes.cz")
        edges = cz["edges"]
        if not isinstance(edges, list):
            _fail("calibration.recipes.cz.edges", "must be an array")
        expected_edges = {
            frozenset(coupling.subsystem_ids) for coupling in model.couplings
        }
        found_edges: set[frozenset[str]] = set()
        required = {
            "subsystems",
            "detuning_subsystem",
            "duration_ns",
            "ramp_duration_ns",
            "detuning_ghz",
        }
        for ordinal, raw in enumerate(edges):
            path = f"calibration.recipes.cz.edges[{ordinal}]"
            edge = _mapping(raw, path)
            _exact_keys(edge, required, path)
            endpoints = edge["subsystems"]
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                _fail(f"{path}.subsystems", "must name exactly two subsystem ids")
            first = _string(endpoints[0], f"{path}.subsystems[0]")
            second = _string(endpoints[1], f"{path}.subsystems[1]")
            edge_key = frozenset((first, second))
            if (
                first == second
                or edge_key not in expected_edges
                or edge_key in found_edges
            ):
                _fail(
                    f"{path}.subsystems", "must name one unique declared coupling edge"
                )
            found_edges.add(edge_key)
            if edge["detuning_subsystem"] not in (first, second):
                _fail(
                    f"{path}.detuning_subsystem", "must be one endpoint of its coupling"
                )
            duration = _number(
                edge["duration_ns"], f"{path}.duration_ns", positive=True
            )
            ramp = _number(
                edge["ramp_duration_ns"], f"{path}.ramp_duration_ns", positive=True
            )
            # Exchange starts the instant the ramp-up ends and runs until the
            # ramp-down begins (parked interval = duration - 2*ramp), so ramp
            # must fit twice within the block with room left for the park.
            if 2 * ramp >= duration:
                _fail(path, "has inconsistent duration and ramp_duration_ns")
            # The first binding uses a near-resonant frame per subsystem, so
            # Delta_i = Delta_j = 0 and the nominal park is -alpha_i. Stored
            # values remain calibrated data and may tune away from that
            # nominal crossing; validation therefore checks the value rather
            # than requiring exact equality to the formula.
            _number(edge["detuning_ghz"], f"{path}.detuning_ghz")

    @staticmethod
    def _validate_single_qubit(
        value: Any,
        path: str,
        expected: set[str],
        *,
        positive_fields: set[str] | None = None,
    ) -> None:
        """Validate one exact scalar-only single-qubit recipe object."""
        data = _mapping(value, path)
        _exact_keys(data, expected, path)
        positive_fields = expected if positive_fields is None else positive_fields
        for field_name in expected:
            _number(
                data[field_name],
                f"{path}.{field_name}",
                positive=field_name in positive_fields,
            )


def load_calibration_spec(document: Any, model: SCTransmonModel) -> CalibrationSpec:
    """Parse and validate a calibration document against its physics model.

    The document must use the versioned ``fatqat.calibration`` envelope and
    repeat the exact builder, model ID, and model revision of ``model``. Recipe
    fields and coupling coverage are checked before an immutable calibration
    is returned.

    Args:
        document: Mapping decoded from a calibration JSON document.
        model: Model returned by :func:`load_physics_model`.

    Returns:
        An immutable, identity-bound :class:`CalibrationSpec`.

    Raises:
        BackendValidationError: If identity, schema, recipe values, or CZ edge
            coverage do not match ``model``.
    """
    return CalibrationSpec.from_mapping(document, model)
