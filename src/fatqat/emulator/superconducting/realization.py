"""Build standard transmon gate-to-pulse maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from ... import operations as ops
from ..._pulse_values import ControlChannel, PulseControl
from ...errors import BackendValidationError
from ...operations.base import Operation
from ...operations.parametric_gates import RY
from ..._waveforms import SampledWaveform
from .._core.target import Frame
from .._core.pulse import PhaseShift, PhaseSwap, PulseDefinition, PulseImplementationMap
from .._core.value_validation import _finite
from .calibration import TransmonCalibration, _CzRecipe
from .model import TransmonModel, angular_rate_from_ghz

_WAVEFORM_SAMPLES = 129
_REALIZATION_CONTRACT = "fixed-qutrit-effective-rwa-v1"

_CanonicalEdge = tuple[str, str]


@dataclass(frozen=True)
class _DragContext:
    alpha: float
    drive: ControlChannel
    frame: Frame
    duration: float
    coefficient: float


@dataclass(frozen=True, slots=True)
class _TransmonMapCompatibility:
    """Private physical-value requirements captured by a standard map."""

    realization_contract: str
    subsystem_anharmonicities: tuple[tuple[str, float], ...]
    control_edges: tuple[_CanonicalEdge, ...]
    cz_detuned_subsystems: tuple[tuple[_CanonicalEdge, str], ...]


class _TransmonPulseImplementationMap(PulseImplementationMap):
    """Standard map carrying destination-model compatibility requirements."""

    def __init__(self, compatibility: _TransmonMapCompatibility) -> None:
        super().__init__()
        self._transmon_compatibility = compatibility

    def copy(self) -> "_TransmonPulseImplementationMap":
        """Return an independent map while retaining compatibility facts."""
        clone = _TransmonPulseImplementationMap(self._transmon_compatibility)
        clone._registry = self._registry.copy()
        return clone


def _canonical_model_edges(model: TransmonModel) -> tuple[_CanonicalEdge, ...]:
    return tuple(
        sorted(tuple(sorted(coupling.subsystem_ids)) for coupling in model._couplings)
    )


def _map_compatibility(
    model: TransmonModel,
    cz_detuned_subsystems: tuple[tuple[_CanonicalEdge, str], ...],
) -> _TransmonMapCompatibility:
    subsystem_anharmonicities = tuple(
        sorted(
            (subsystem.id, subsystem.anharmonicity_ghz)
            for subsystem in model._subsystems
        )
    )
    return _TransmonMapCompatibility(
        _REALIZATION_CONTRACT,
        subsystem_anharmonicities,
        _canonical_model_edges(model),
        tuple(sorted(cz_detuned_subsystems)),
    )


def _compatibility_mismatch(
    source: _TransmonMapCompatibility, model: TransmonModel
) -> str | None:
    if source.realization_contract != _REALIZATION_CONTRACT:
        return (
            f"realization contract changed from {source.realization_contract!r} "
            f"to {_REALIZATION_CONTRACT!r}"
        )

    source_anharmonicities = dict(source.subsystem_anharmonicities)
    destination_anharmonicities = {
        subsystem.id: subsystem.anharmonicity_ghz for subsystem in model._subsystems
    }
    source_labels = set(source_anharmonicities)
    destination_labels = set(destination_anharmonicities)
    if source_labels != destination_labels:
        missing = sorted(source_labels - destination_labels)
        unexpected = sorted(destination_labels - source_labels)
        return (
            f"subsystem labels changed (missing={missing!r}, unexpected={unexpected!r})"
        )

    selected_edges = {selected: edge for edge, selected in source.cz_detuned_subsystems}
    for label, expected in source.subsystem_anharmonicities:
        actual = destination_anharmonicities[label]
        if actual != expected:
            branch_context = (
                f" selected by CZ edge {selected_edges[label]!r}"
                if label in selected_edges
                else ""
            )
            return (
                f"anharmonicity for subsystem {label!r}{branch_context} changed "
                f"from {expected!r} to {actual!r}"
            )

    destination_edges = _canonical_model_edges(model)
    if source.control_edges != destination_edges:
        source_edges = set(source.control_edges)
        destination_edge_set = set(destination_edges)
        missing = sorted(source_edges - destination_edge_set)
        unexpected = sorted(destination_edge_set - source_edges)
        return f"canonical topology changed (missing={missing!r}, unexpected={unexpected!r})"
    return None


def _validate_transmon_map_compatibility(
    implementations: object, model: TransmonModel
) -> None:
    """Reject a standard compiled map whose captured model facts changed."""
    if not isinstance(implementations, _TransmonPulseImplementationMap):
        return
    source = implementations._transmon_compatibility
    mismatch = _compatibility_mismatch(source, model)
    if mismatch is not None:
        raise BackendValidationError(
            "compiled transmon gate map is incompatible with this model: "
            f"{mismatch}; "
            "rebuild it with default_transmon_gate_implementation_map()"
        )


def _sample_grid(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, _WAVEFORM_SAMPLES)


def _hann(tlist: np.ndarray, duration: float, peak: float) -> np.ndarray:
    return peak * np.sin(pi * tlist / duration) ** 2


def _cumulative_trapezoid(values: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(values, dtype=float)
    phase[1:] = np.cumsum((values[1:] + values[:-1]) * np.diff(tlist) / 2.0)
    return phase


def _drag_definition(operation: Operation, context: _DragContext) -> PulseDefinition:
    theta = _finite(operation.theta, "rotation angle")
    tlist = _sample_grid(context.duration)
    p = _hann(tlist, context.duration, theta / context.duration)
    dp = theta * pi * np.sin(2 * pi * tlist / context.duration) / context.duration**2
    x0 = p - p**3 / context.alpha**2
    zv = -2 * p**2 / context.alpha
    y0 = -context.coefficient * dp / (context.alpha + zv)
    phase = _cumulative_trapezoid(zv, tlist)
    envelope = (x0 + 1j * y0) * np.exp(1j * phase)
    if isinstance(operation, RY):
        envelope *= 1j
    return PulseDefinition(
        context.duration,
        (PulseControl(context.drive, SampledWaveform(tlist, envelope)),),
        (PhaseShift(context.frame, float(phase[-1])),),
    )


def _iswap_definition(
    model: TransmonModel,
    calibration: TransmonCalibration,
    first: str,
    second: str,
) -> PulseDefinition:
    duration = calibration._iswap_duration_ns
    tlist = _sample_grid(duration)
    exchange = _hann(tlist, duration, -pi / duration)
    return PulseDefinition(
        duration,
        (
            PulseControl(
                model.control.exchange(first, second),
                SampledWaveform(tlist, exchange),
            ),
        ),
        (PhaseSwap(model.frame(first), model.frame(second)),),
    )


def _cz_definition(
    model: TransmonModel,
    recipe: _CzRecipe,
    first: str,
    second: str,
) -> PulseDefinition:
    duration = recipe.duration_ns
    ramp = recipe.ramp_duration_ns
    parked_duration = duration - 2 * ramp
    detuning_grid = _sample_grid(duration)
    ramp_shape = np.ones_like(detuning_grid)
    if ramp > 0:
        rising = detuning_grid < ramp
        falling = detuning_grid > duration - ramp
        ramp_shape[rising] = (1 - np.cos(pi * detuning_grid[rising] / ramp)) / 2
        ramp_shape[falling] = (
            1 - np.cos(pi * (duration - detuning_grid[falling]) / ramp)
        ) / 2
    detuning = angular_rate_from_ghz(recipe.park_detuning_ghz) * ramp_shape
    exchange_grid = _sample_grid(parked_duration)
    exchange = _hann(exchange_grid, parked_duration, sqrt(2) * pi / parked_duration)
    detuning_subsystem = recipe.detuned_subsystem
    detuning_phase = float(_cumulative_trapezoid(detuning, detuning_grid)[-1])
    return PulseDefinition(
        duration,
        (
            PulseControl(
                model.control.detuning(detuning_subsystem),
                SampledWaveform(detuning_grid, detuning),
            ),
            PulseControl(
                model.control.exchange(first, second),
                SampledWaveform(exchange_grid, exchange),
                ramp,
            ),
        ),
        (PhaseShift(model.frame(detuning_subsystem), detuning_phase),),
    )


def _resolve_cz_recipes(
    model: TransmonModel, calibration: TransmonCalibration
) -> dict[_CanonicalEdge, _CzRecipe]:
    subsystems = {subsystem.id: subsystem for subsystem in model._subsystems}
    resolved: dict[_CanonicalEdge, _CzRecipe] = {}
    for edge in _canonical_model_edges(model):
        recipe = calibration._cz_recipe(*edge)
        if recipe is None:
            raise BackendValidationError(
                f"calibration has no CZ recipe for canonical model edge {edge!r}"
            )
        selected = subsystems[recipe.detuned_subsystem]
        branch_error = abs(recipe.park_detuning_ghz + selected.anharmonicity_ghz)
        if branch_error > recipe.branch_tolerance_ghz:
            expected = -selected.anharmonicity_ghz
            raise BackendValidationError(
                f"CZ recipe for edge {edge!r} selects {selected.id!r} with "
                f"park_detuning_ghz={recipe.park_detuning_ghz!r}; expected "
                f"{expected!r} within branch_tolerance_ghz="
                f"{recipe.branch_tolerance_ghz!r}"
            )
        resolved[edge] = recipe
    return resolved


def default_transmon_gate_implementation_map(
    *, model: TransmonModel, calibration: TransmonCalibration
) -> PulseImplementationMap:
    """Build the standard transmon gate-to-pulse map.

    The returned rules use the model's channels, frames, subsystem parameters,
    and coupling graph together with the supplied calibration. Rebuild the map
    after changing values that should alter those pulse recipes.

    Args:
        model: Model whose physical resources and parameters the rules use.
        calibration: Gate recipe values.

    Returns:
        A new map for ``RX``, ``RY``, ``RZ``, ``iSwap``, and coupled ``CZ``.

    Raises:
        BackendValidationError: If either argument has the wrong type, a model
            edge has no CZ recipe, or a selected CZ branch is incompatible
            with the model's signed anharmonicity.
    """
    if not isinstance(model, TransmonModel):
        raise BackendValidationError("model must be a TransmonModel")
    if not isinstance(calibration, TransmonCalibration):
        raise BackendValidationError("calibration must be a TransmonCalibration")

    cz_recipes = _resolve_cz_recipes(model, calibration)
    drag_contexts = {
        subsystem.id: _DragContext(
            angular_rate_from_ghz(subsystem.anharmonicity_ghz),
            model.control.drive(subsystem.id),
            model.frame(subsystem.id),
            calibration._rx_ry_duration_ns,
            calibration._rx_ry_drag_coefficient,
        )
        for subsystem in model._subsystems
    }
    rz_frames = {
        subsystem.id: model.frame(subsystem.id) for subsystem in model._subsystems
    }

    def rx_ry(operation: Operation, *, device_operands: tuple[object, ...]):
        try:
            context = drag_contexts[device_operands[0]]
        except (IndexError, KeyError):
            raise BackendValidationError(
                f"compiled transmon map has no subsystem {device_operands!r}"
            ) from None
        return _drag_definition(operation, context)

    def rz(operation: Operation, *, device_operands: tuple[object, ...]):
        try:
            frame = rz_frames[device_operands[0]]
        except (IndexError, KeyError):
            raise BackendValidationError(
                f"compiled transmon map has no subsystem {device_operands!r}"
            ) from None
        return PulseDefinition(
            0.0,
            (),
            (PhaseShift(frame, _finite(operation.theta, "rotation angle")),),
        )

    cz_detuned_subsystems = tuple(
        (edge, recipe.detuned_subsystem) for edge, recipe in cz_recipes.items()
    )
    implementations = _TransmonPulseImplementationMap(
        _map_compatibility(model, cz_detuned_subsystems)
    )
    implementations.add(ops.RX, rx_ry)
    implementations.add(ops.RY, rx_ry)
    implementations.add(ops.RZ, rz)
    for coupling in model._couplings:
        first, second = coupling.subsystem_ids
        canonical_edge = tuple(sorted((first, second)))
        cz_definition = _cz_definition(
            model, cz_recipes[canonical_edge], *canonical_edge
        )
        for ordered in ((first, second), (second, first)):
            implementations.add(
                ops.iSwap,
                _iswap_definition(model, calibration, *ordered),
                device_operands=ordered,
            )
            implementations.add(
                ops.CZ,
                cz_definition,
                device_operands=ordered,
            )
    return implementations
