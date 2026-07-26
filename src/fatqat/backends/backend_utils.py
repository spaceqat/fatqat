"""Shared backend utilities that are representation-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .._engine_index_allocation import _EngineIndexAllocation
from ..errors import BackendValidationError
from ..noise import NoiseModel
from ..operations.measurement import Measurement
from ..registers import RegisterRef
from ..resource_layout import ResourceLayout


def _validate_grid_size(grid_size: object) -> tuple[int, int]:
    """Return a validated ``(rows, columns)`` grid shape."""
    if not isinstance(grid_size, tuple):
        raise TypeError(
            "grid_size must be a tuple of two positive integers, got "
            f"{type(grid_size)!r}"
        )
    if len(grid_size) != 2:
        raise ValueError("grid_size must contain exactly two values: (rows, columns)")
    rows, cols = grid_size
    for name, value in (("grid_size[0]", rows), ("grid_size[1]", cols)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value)!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return rows, cols


@dataclass(frozen=True)
class _LoweringContext:
    """Private per-run pairing of the resolved `ResourceLayout` and `_EngineIndexAllocation`.

    A lowering-time implementation convenience only: it is not public and it
    does not combine one program ref's device label and engine index into a
    single per-resource value (contrast the removed `BoundResource`, which
    did exactly that). Matrix lowering reads `resource_layout` to build
    `ImplementationMap` lookup keys (`device_operands`) and reads
    `engine_index_allocation` for every execution-plan index/dimension
    (`ApplyMatrixStep` targets, measurement, reset, and condition lowering).
    Both values are resolved once per run and threaded through unchanged;
    neither is re-resolved during lowering.
    """

    resource_layout: ResourceLayout
    engine_index_allocation: _EngineIndexAllocation


@dataclass(frozen=True)
class _PlanFacts:
    """Program facts needed for backend result defaults and validation.

    ``has_channel`` defaults to ``False`` so channel-free construction sites
    (and the fake-backend lowering) stay unchanged.
    """

    has_measurement: bool
    has_reset: bool
    has_channel: bool = False


def _normalize_config(
    config: dict[str, Any] | None,
    config_cls: type,
    param_name: str,
    backend_name: str = "SimulatorBackend",
) -> Any:
    """Strictly normalize a plain dictionary into a frozen config dataclass.

    Supported keys are derived from ``config_cls``'s dataclass fields, so a
    newly added configuration entry needs no registry update. Field-level
    compatibility is owned by the config dataclass's ``__post_init__``;
    subclasses are responsible for validating fields they add.
    """
    if config is None:
        return config_cls()
    if not isinstance(config, dict):
        raise TypeError(f"{param_name} must be a dict or None, got {type(config)!r}")
    supported = {field.name for field in fields(config_cls)}
    unknown = set(config) - supported
    if unknown:
        # Dict keys need not be mutually orderable (for example, ``"foo"``
        # and ``3``). Sort their representations so malformed input still
        # receives our validation error rather than a Python ``TypeError``.
        names = ", ".join(sorted(map(repr, unknown)))
        expected = ", ".join(repr(name) for name in sorted(supported)) or "no keys"
        raise BackendValidationError(
            f"{backend_name} does not support {param_name} key(s) {names}; "
            f"expected {expected}"
        )
    return config_cls(**config)


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    engine_index_allocation: _EngineIndexAllocation,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    return tuple(
        (engine_index_allocation.clbit_index(ref), val) for ref, val in condition
    )


def _resolve_confusions(
    measured_targets: tuple[RegisterRef, ...],
    measured_indices: tuple[int, ...],
    reported_digit_maps: tuple[tuple[int, ...], ...],
    resource_layout: ResourceLayout,
    noise_model: NoiseModel,
) -> tuple[Any, ...] | None:
    """Resolve per-subsystem readout confusion matrices for one measurement.

    ``readout_error_for`` is the single source of truth per subsystem; this
    function only collapses an all-``None`` resolution back to ``None`` so
    the noise-free (and the common) case allocates nothing on the step.
    Selection matches against each measured ref's logical identity and/or
    resource-layout device label (never an engine index); the paired engine
    index is used only in the error message, never derived backward from a
    device label.

    Shared by the matrix and pulse families: each caller supplies its own
    ``reported_digit_maps`` (matrix's per-subsystem identity range, pulse's
    literal qutrit-to-bit map), and this validates a selected confusion
    matrix's shape against the reported dimension implied by that map
    (``max(reported_map) + 1``) - it does not also require the map's length
    to equal any subsystem's physical Hilbert dimension, since pulse's
    program-declared (qubit) dimension and its simulated (qutrit) dimension
    deliberately differ.

    Raises :py:exc:`~fatqat.errors.BackendValidationError` if a selected
    matrix's dimension does not match the reported classical digit dimension.
    """
    resolved = []
    for target, measured, reported_map in zip(
        measured_targets, measured_indices, reported_digit_maps
    ):
        confusion = noise_model.readout_error_for(target, resource_layout)
        if confusion is not None:
            reported_dim = max(reported_map) + 1
            if confusion.shape != (reported_dim, reported_dim):
                raise BackendValidationError(
                    f"readout confusion matrix of shape {confusion.shape} "
                    f"selected for subsystem {measured} has reported classical "
                    f"dimension {reported_dim}"
                )
        resolved.append(confusion)
    if all(confusion is None for confusion in resolved):
        return None
    return tuple(resolved)


def _lower_measurement_boundary(
    step: Measurement,
    reported_digit_maps: tuple[tuple[int, ...], ...],
    resource_layout: ResourceLayout,
    engine_index_allocation: _EngineIndexAllocation,
    noise_model: NoiseModel,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[Any, ...] | None]:
    """Resolve the measurement-lowering boundary shared by both backend families.

    Returns ``(measured_indices, classical_indices, confusions)``. The caller
    decides what to store as `MeasurementStep.reported_digit_maps`: matrix's
    noise-free identity case must keep passing ``None`` (the compatibility
    default a numba-compiled fast path recognizes), while pulse always
    passes its literal qutrit-to-bit map. This function only resolves engine
    indices and validates/collapses readout confusion against the caller's
    supplied map; it never decides the step's stored map itself.
    """
    measured_indices = tuple(
        engine_index_allocation.subsystem_index(target) for target in step.targets
    )
    classical_indices = tuple(
        engine_index_allocation.clbit_index(output) for output in step.outputs
    )
    confusions = _resolve_confusions(
        step.targets,
        measured_indices,
        reported_digit_maps,
        resource_layout,
        noise_model,
    )
    return measured_indices, classical_indices, confusions
