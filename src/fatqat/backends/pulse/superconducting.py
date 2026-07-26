"""Data-only superconducting transmon/exchange model foundations.

This module deliberately contains no solver, global tensor operator, or QuTiP
object.  It validates a durable model snapshot, builds immutable local-qutrit
facts and opaque model handles, and loads a separately identity-bound
calibration profile for the later pulse-realization layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any

import numpy as np

from ...errors import BackendValidationError

_MODEL_FORMAT = "fatqat.physics-model"
_CALIBRATION_FORMAT = "fatqat.calibration"
_SC_BUILDER_ID = "sc.transmon_exchange"
_SC_BUILDER_VERSION = 1
_SCHEMA_VERSION = 1


def _fail(path: str, message: str) -> None:
    raise BackendValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(path, "must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
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
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _version(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        _fail(path, "must be a positive integer")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not isfinite(result):
        _fail(path, "must be finite")
    if positive and result <= 0:
        _fail(path, "must be positive")
    return result


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=complex, copy=True)
    array.flags.writeable = False
    return array


def _freeze_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_data(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_data(child) for child in value)
    return value


@dataclass(frozen=True)
class BuilderIdentity:
    """Trusted builder key persisted in model and calibration documents."""

    id: str
    version: int

    @classmethod
    def from_mapping(cls, value: Any, path: str = "builder") -> BuilderIdentity:
        data = _mapping(value, path)
        _exact_keys(data, {"id", "version"}, path)
        return cls(
            id=_string(data["id"], f"{path}.id"),
            version=_version(data["version"], f"{path}.version"),
        )


@dataclass(frozen=True)
class ModelIdentity:
    """Stable identity for one immutable persisted model snapshot."""

    id: str
    revision: str

    @classmethod
    def from_mapping(cls, value: Any, path: str = "model") -> ModelIdentity:
        data = _mapping(value, path)
        _exact_keys(data, {"id", "revision"}, path)
        return cls(
            id=_string(data["id"], f"{path}.id"),
            revision=_string(data["revision"], f"{path}.revision"),
        )


@dataclass(frozen=True)
class ModelKey:
    """Complete model identity used to bind opaque handles and calibration."""

    builder: BuilderIdentity
    model: ModelIdentity


@dataclass(frozen=True)
class PhysicsModelSpec:
    """Canonical data-only, versioned physics-model persistence envelope."""

    builder: BuilderIdentity
    model: ModelIdentity
    parameters: Mapping[str, Any]
    format: str = _MODEL_FORMAT
    schema_version: int = _SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, document: Any) -> PhysicsModelSpec:
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
        return ModelKey(self.builder, self.model)


@dataclass(frozen=True)
class Transmon:
    """One model-local fixed-qutrit transmon's durable numerical facts."""

    id: str
    frequency_ghz: float
    anharmonicity_ghz: float


@dataclass(frozen=True)
class Coupling:
    """One undirected, effective-frame exchange edge."""

    id: str
    subsystem_ids: tuple[str, str]
    residual_exchange_ghz: float


@dataclass(frozen=True)
class SubsystemResourceRef:
    """Opaque physical subsystem resource minted by one :class:`PhysicsModel`."""

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ControlChannelRef:
    """Opaque complex-drive control channel minted by one model."""

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class FrameRef:
    """Opaque virtual-frame handle minted by one model."""

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class CouplingRef:
    """Opaque exchange-resource handle minted by one model."""

    model_key: ModelKey
    ordinal: int
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicsModel:
    """Immutable engine-neutral SC model; it contains local facts only."""

    key: ModelKey
    subsystems: tuple[Transmon, ...]
    couplings: tuple[Coupling, ...]
    annihilation: np.ndarray
    creation: np.ndarray
    number: np.ndarray
    _token: object = field(repr=False, compare=False)
    _resources: tuple[SubsystemResourceRef, ...] = field(repr=False)
    _controls: tuple[ControlChannelRef, ...] = field(repr=False)
    _frames: tuple[FrameRef, ...] = field(repr=False)
    _coupling_refs: tuple[CouplingRef, ...] = field(repr=False)

    @property
    def time_unit(self) -> str:
        """Canonical runtime coordinate for this model family."""
        return "ns"

    @property
    def subsystem_ids(self) -> tuple[str, ...]:
        return tuple(subsystem.id for subsystem in self.subsystems)

    @property
    def physical_dimension(self) -> int:
        return 3

    def resource(self, subsystem_id: str) -> SubsystemResourceRef:
        return self._resources[self._subsystem_ordinal(subsystem_id)]

    def drive_control(self, subsystem_id: str) -> ControlChannelRef:
        return self._controls[self._subsystem_ordinal(subsystem_id)]

    def frame(self, subsystem_id: str) -> FrameRef:
        return self._frames[self._subsystem_ordinal(subsystem_id)]

    def coupling(self, first: str, second: str) -> CouplingRef:
        edge = frozenset((first, second))
        for ordinal, coupling in enumerate(self.couplings):
            if frozenset(coupling.subsystem_ids) == edge:
                return self._coupling_refs[ordinal]
        raise BackendValidationError(
            f"model has no declared coupling edge {first!r}-{second!r}"
        )

    def bind_resource(self, reference: SubsystemResourceRef) -> int:
        return self._bind(reference, SubsystemResourceRef, self._resources, "resource")

    def bind_control(self, reference: ControlChannelRef) -> int:
        return self._bind(reference, ControlChannelRef, self._controls, "control")

    def bind_frame(self, reference: FrameRef) -> int:
        return self._bind(reference, FrameRef, self._frames, "frame")

    def bind_coupling(self, reference: CouplingRef) -> int:
        return self._bind(reference, CouplingRef, self._coupling_refs, "coupling")

    def _subsystem_ordinal(self, subsystem_id: str) -> int:
        try:
            return self.subsystem_ids.index(subsystem_id)
        except ValueError:
            raise BackendValidationError(
                f"unknown model subsystem {subsystem_id!r}"
            ) from None

    def _bind(
        self, reference: Any, kind: type, refs: tuple[Any, ...], name: str
    ) -> int:
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
    """Registry containing only package-trusted physics-model builders."""

    def __init__(self) -> None:
        self._builders: dict[BuilderIdentity, SCTransmonExchangeBuilder] = {}

    def register(self, builder: SCTransmonExchangeBuilder) -> None:
        identity = builder.identity
        if identity in self._builders:
            raise ValueError(
                f"builder already registered: {identity.id}/{identity.version}"
            )
        self._builders[identity] = builder

    def resolve(self, identity: BuilderIdentity) -> SCTransmonExchangeBuilder:
        try:
            return self._builders[identity]
        except KeyError:
            raise BackendValidationError(
                f"unsupported physics-model builder {identity.id!r} version {identity.version}"
            ) from None

    def build(self, spec: PhysicsModelSpec) -> PhysicsModel:
        return self.resolve(spec.builder).build(spec)


class SCTransmonExchangeBuilder:
    """Trusted builder for fixed-qutrit, arbitrary-graph SC exchange models."""

    identity = BuilderIdentity(_SC_BUILDER_ID, _SC_BUILDER_VERSION)

    def build(self, spec: PhysicsModelSpec) -> PhysicsModel:
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
        annihilation = _freeze_array(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, sqrt(2)], [0.0, 0.0, 0.0]])
        )
        creation = _freeze_array(annihilation.conj().T)
        number = _freeze_array(np.diag([0.0, 1.0, 2.0]))
        token = object()
        resources = tuple(
            SubsystemResourceRef(spec.key, ordinal, token)
            for ordinal in range(len(subsystems))
        )
        controls = tuple(
            ControlChannelRef(spec.key, ordinal, token)
            for ordinal in range(len(subsystems))
        )
        frames = tuple(
            FrameRef(spec.key, ordinal, token) for ordinal in range(len(subsystems))
        )
        coupling_refs = tuple(
            CouplingRef(spec.key, ordinal, token) for ordinal in range(len(couplings))
        )
        return PhysicsModel(
            key=spec.key,
            subsystems=subsystems,
            couplings=couplings,
            annihilation=annihilation,
            creation=creation,
            number=number,
            _token=token,
            _resources=resources,
            _controls=controls,
            _frames=frames,
            _coupling_refs=coupling_refs,
        )

    @staticmethod
    def _validate_units(value: Any) -> None:
        units = _mapping(value, "physics model.parameters.units")
        _exact_keys(
            units, {"subsystems", "couplings"}, "physics model.parameters.units"
        )
        subsystem_units = _mapping(
            units["subsystems"], "physics model.parameters.units.subsystems"
        )
        _exact_keys(
            subsystem_units,
            {"frequency", "anharmonicity"},
            "physics model.parameters.units.subsystems",
        )
        coupling_units = _mapping(
            units["couplings"], "physics model.parameters.units.couplings"
        )
        _exact_keys(
            coupling_units,
            {"residual_exchange"},
            "physics model.parameters.units.couplings",
        )
        if (
            subsystem_units["frequency"] != "GHz"
            or subsystem_units["anharmonicity"] != "GHz"
        ):
            _fail(
                "physics model.parameters.units.subsystems",
                "frequency and anharmonicity must use GHz",
            )
        if coupling_units["residual_exchange"] != "GHz":
            _fail(
                "physics model.parameters.units.couplings.residual_exchange",
                "must use GHz",
            )

    @staticmethod
    def _build_subsystems(value: Any) -> tuple[Transmon, ...]:
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
        if not isinstance(value, list):
            _fail("physics model.parameters.couplings", "must be an array")
        known = {subsystem.id for subsystem in subsystems}
        ids: set[str] = set()
        edges: set[frozenset[str]] = set()
        couplings = []
        for ordinal, raw in enumerate(value):
            path = f"physics model.parameters.couplings[{ordinal}]"
            item = _mapping(raw, path)
            if set(item) - {"id", "subsystems", "residual_exchange"} or {
                "id",
                "subsystems",
            } - set(item):
                _fail(
                    path,
                    "must contain id and subsystems, with optional residual_exchange",
                )
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
            residual = _number(
                item.get("residual_exchange", 0.0), f"{path}.residual_exchange"
            )
            couplings.append(Coupling(identifier, (first, second), residual))
        return tuple(couplings)


PHYSICS_MODEL_BUILDERS = PhysicsModelBuilderRegistry()
PHYSICS_MODEL_BUILDERS.register(SCTransmonExchangeBuilder())


def load_physics_model(document: Any) -> PhysicsModel:
    """Parse and build one trusted physics-model document."""
    return PHYSICS_MODEL_BUILDERS.build(PhysicsModelSpec.from_mapping(document))


@dataclass(frozen=True)
class CalibrationSpec:
    """Immutable calibration recipe data bound to exactly one model snapshot."""

    key: ModelKey
    recipes: Mapping[str, Any]
    format: str = _CALIBRATION_FORMAT
    schema_version: int = _SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, document: Any, model: PhysicsModel) -> CalibrationSpec:
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
        try:
            return self.recipes[name]
        except KeyError:
            raise BackendValidationError(
                f"unknown calibration recipe {name!r}"
            ) from None

    @staticmethod
    def _validate_recipes(recipes: Mapping[str, Any], model: PhysicsModel) -> None:
        _exact_keys(recipes, {"rx_ry", "rz", "iswap", "cz"}, "calibration.recipes")
        # No stored amplitude: for a requested angle theta and this duration T,
        # the Hann/DRAG peak is exactly theta/T (architecture doc Sec. 5.1), so
        # realization derives it rather than reading a calibrated constant.
        CalibrationSpec._validate_single_qubit(
            recipes["rx_ry"],
            "calibration.recipes.rx_ry",
            {"duration_ns", "drag_coefficient"},
        )
        CalibrationSpec._validate_single_qubit(
            recipes["rz"],
            "calibration.recipes.rz",
            {"frame_scale"},
            positive_fields=set(),
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
            "phase_corrections_rad",
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
            # `detuning_ghz` is the calibrated park point bringing |11> into
            # resonance with |20>/|02>: delta_park = -(Delta_i - Delta_j +
            # alpha_i) (architecture doc Sec. 5.3), where alpha_i is
            # detuning_subsystem's anharmonicity. Delta_i/Delta_j are each
            # subsystem's coefficient in the model's rotating-frame H_drift,
            # and Sec. 4 states these are frame-convention dependent - not
            # simply each subsystem's raw `frequency_ghz`. Whether the
            # binding uses one shared reference frame (Delta_i - Delta_j
            # reduces to the raw frequency split) or gives each subsystem
            # its own near-resonant frame (Delta_i - Delta_j ~ 0, so
            # delta_park ~ -alpha_i) is not yet pinned down anywhere in the
            # design docs; see spec Sec. 10 "detuning_ghz frame convention".
            # Values stored here are calibration data, not derived by this
            # module, so this ambiguity does not block validation - it only
            # means a value here should not be trusted as physically exact
            # until the frame convention is fixed and this field is
            # recomputed against it.
            _number(edge["detuning_ghz"], f"{path}.detuning_ghz")
            corrections = _mapping(
                edge["phase_corrections_rad"], f"{path}.phase_corrections_rad"
            )
            if set(corrections) != {first, second}:
                _fail(
                    f"{path}.phase_corrections_rad",
                    "must contain exactly both edge endpoints",
                )
            for subsystem_id, phase in corrections.items():
                _number(phase, f"{path}.phase_corrections_rad.{subsystem_id}")
        if found_edges != expected_edges:
            _fail(
                "calibration.recipes.cz.edges",
                "must provide one recipe for every declared coupling edge",
            )

    @staticmethod
    def _validate_single_qubit(
        value: Any,
        path: str,
        expected: set[str],
        *,
        positive_fields: set[str] | None = None,
    ) -> None:
        data = _mapping(value, path)
        _exact_keys(data, expected, path)
        positive_fields = expected if positive_fields is None else positive_fields
        for field_name in expected:
            _number(
                data[field_name],
                f"{path}.{field_name}",
                positive=field_name in positive_fields,
            )


def load_calibration_spec(document: Any, model: PhysicsModel) -> CalibrationSpec:
    """Parse and validate one calibration document against its physics model."""
    return CalibrationSpec.from_mapping(document, model)
