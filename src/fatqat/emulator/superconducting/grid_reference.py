"""Deterministic reference documents for rectangular transmon grids."""

from __future__ import annotations

import hashlib
import json
import warnings
from math import isfinite
from typing import Any

import numpy as np

_GENERATOR_VERSION = 1
_DRAW_DOMAIN = f"fatqat.transmon-grid.v{_GENERATOR_VERSION}"
_MODEL_ID = "synthetic-transmon-grid-reference"
_CALIBRATION_ID = "fatqat_generated_transmon_grid_reference"

_RX_RY_DURATION_NS = 20.0
_DRAG_COEFFICIENT = 1.0
_ISWAP_DURATION_NS = 40.0
_CZ_DURATION_NS = 60.0
_CZ_RAMP_DURATION_NS = 3.0
_CZ_BRANCH_TOLERANCE_GHZ = 1e-12
_WARNING_EDGE_LABEL_LIMIT = 10

_CanonicalEdge = tuple[str, str]


def _frequency_value(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a built-in int or float, excluding bool")
    try:
        normalized = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_shape(shape: object) -> tuple[int, int]:
    if type(shape) is not tuple:
        raise TypeError("shape must be a tuple")
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    rows, columns = shape
    if type(rows) is not int or type(columns) is not int:
        raise TypeError("shape dimensions must be exact non-Boolean integers")
    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive")
    if rows * columns < 2:
        raise ValueError("shape must contain at least two sites")
    return rows, columns


def _validate_frequency_groups(value: object) -> tuple[float, float]:
    if type(value) is not tuple:
        raise TypeError("frequency_groups_ghz must be a tuple")
    if len(value) != 2:
        raise ValueError("frequency_groups_ghz must contain exactly two values")
    first = _frequency_value(value[0], "frequency_groups_ghz[0]")
    second = _frequency_value(value[1], "frequency_groups_ghz[1]")
    if first <= 0 or second <= 0:
        raise ValueError("frequency group centers must be positive")
    if first == second:
        raise ValueError("frequency group centers must be distinct")
    return first, second


def _site_normal_draw(seed: int, label: str) -> float:
    material = f"{_DRAW_DOMAIN}\0{seed}\0{label}\0frequency".encode("utf-8")
    site_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return float(np.random.Generator(np.random.PCG64(site_seed)).standard_normal())


def _canonical_revision(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _edge_sequence(rows: int, columns: int) -> list[_CanonicalEdge]:
    edges: list[_CanonicalEdge] = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            label = f"q{index}"
            if column + 1 < columns:
                edges.append(tuple(sorted((label, f"q{index + 1}"))))
            if row + 1 < rows:
                edges.append(tuple(sorted((label, f"q{index + columns}"))))
    return edges


def _warn_about_realized_ordering(
    *,
    group_frequencies: tuple[list[float], list[float]],
    edges: list[_CanonicalEdge],
    frequencies: dict[str, float],
    groups: dict[str, int],
    centers: tuple[float, float],
) -> None:
    first_range = (min(group_frequencies[0]), max(group_frequencies[0]))
    second_range = (min(group_frequencies[1]), max(group_frequencies[1]))
    ranges_overlap = (
        first_range[1] >= second_range[0] and second_range[1] >= first_range[0]
    )

    reversed_or_tied: list[_CanonicalEdge] = []
    group_zero_should_be_lower = centers[0] < centers[1]
    for edge in edges:
        group_zero_label = edge[0] if groups[edge[0]] == 0 else edge[1]
        group_one_label = edge[1] if group_zero_label == edge[0] else edge[0]
        group_zero_frequency = frequencies[group_zero_label]
        group_one_frequency = frequencies[group_one_label]
        if (
            group_zero_frequency >= group_one_frequency
            if group_zero_should_be_lower
            else group_zero_frequency <= group_one_frequency
        ):
            reversed_or_tied.append(edge)

    if not ranges_overlap and not reversed_or_tied:
        return
    range_status = "touch or overlap" if ranges_overlap else "do not overlap"
    shown_edges = reversed_or_tied[:_WARNING_EDGE_LABEL_LIMIT]
    edge_labels = ", ".join(f"({first}, {second})" for first, second in shown_edges)
    if not edge_labels:
        edge_labels = "none"
    omitted = len(reversed_or_tied) - len(shown_edges)
    if omitted:
        edge_labels += f", ... and {omitted} more"
    warnings.warn(
        "Generated transmon realized group ranges "
        f"{range_status}; {len(reversed_or_tied)} nearest-neighbor edge(s) "
        "reverse or tie the group-center ordering: "
        f"{edge_labels}.",
        UserWarning,
        stacklevel=4,
    )


def _model_document(
    *,
    rows: int,
    columns: int,
    centers: tuple[float, float],
    standard_deviation: float,
    anharmonicity: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float], list[_CanonicalEdge]]:
    labels = [f"q{index}" for index in range(rows * columns)]
    frequencies: dict[str, float] = {}
    groups: dict[str, int] = {}
    group_frequencies: tuple[list[float], list[float]] = ([], [])
    parameters: dict[str, dict[str, float]] = {}

    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            label = labels[index]
            group = (row + column) % 2
            draw = 0.0 if standard_deviation == 0 else _site_normal_draw(seed, label)
            frequency = centers[group] + standard_deviation * draw
            if not isfinite(frequency) or frequency <= 0:
                raise ValueError(
                    f"realized frequency for site {label} must be finite and positive"
                )
            frequencies[label] = frequency
            groups[label] = group
            group_frequencies[group].append(frequency)
            parameters[label] = {
                "frequency": frequency,
                "anharmonicity": anharmonicity,
            }

    edges = _edge_sequence(rows, columns)
    _warn_about_realized_ordering(
        group_frequencies=group_frequencies,
        edges=edges,
        frequencies=frequencies,
        groups=groups,
        centers=centers,
    )
    document: dict[str, Any] = {
        "format": {"id": "sc.transmon_exchange", "version": 1},
        "model": {"id": _MODEL_ID},
        "system": {
            "subsystem_type": "transmon",
            "subsystems": labels,
            "control_edges": [
                {"id": f"e{ordinal}", "subsystems": list(edge)}
                for ordinal, edge in enumerate(edges)
            ],
        },
        "units": {"frequency": "GHz", "anharmonicity": "GHz"},
        "parameters": {"subsystems": parameters},
    }
    document["model"]["revision"] = _canonical_revision(document)
    return document, frequencies, edges


def _calibration_document(
    *,
    edges: list[_CanonicalEdge],
    frequencies: dict[str, float],
    anharmonicity: float,
) -> dict[str, Any]:
    cz_entries = []
    for edge in edges:
        first, second = edge
        selected = first if frequencies[first] >= frequencies[second] else second
        cz_entries.append(
            {
                "canonical_edge": [first, second],
                "recipe": {
                    "detuned_subsystem": selected,
                    "duration": _CZ_DURATION_NS,
                    "ramp_duration": _CZ_RAMP_DURATION_NS,
                    "park_detuning_ghz": -anharmonicity,
                    "branch_tolerance_ghz": _CZ_BRANCH_TOLERANCE_GHZ,
                },
            }
        )

    document: dict[str, Any] = {
        "format": {
            "id": "sc.transmon_exchange_fixed_pulse",
            "version": 1,
        },
        "calibration": {"id": _CALIBRATION_ID},
        "provenance": {
            "kind": "generated_reference_recipe",
            "generator_version": _GENERATOR_VERSION,
            "numerically_calibrated": False,
        },
        "units": {
            "time": "ns",
            "frequency": "GHz",
            "dimensionless": "1",
        },
        "recipes": {
            "rx_ry": {
                "duration": _RX_RY_DURATION_NS,
                "drag_coefficient": _DRAG_COEFFICIENT,
            },
            "iswap": {"duration": _ISWAP_DURATION_NS},
            "cz": {"edges": cz_entries},
        },
    }
    document["calibration"]["revision"] = _canonical_revision(document)
    return document


def generate_transmon_grid_documents(
    *,
    shape: tuple[int, int],
    frequency_groups_ghz: tuple[float, float],
    frequency_std_ghz: float = 0.010,
    anharmonicity_ghz: float = -0.22,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return matching model and calibration documents for a transmon grid.

    This function constructs model and analytic reference-calibration JSON. It
    performs no numerical calibration or simulation. Both returned mappings
    are mutable and owned by the caller.
    """
    rows, columns = _validate_shape(shape)
    centers = _validate_frequency_groups(frequency_groups_ghz)
    standard_deviation = _frequency_value(frequency_std_ghz, "frequency_std_ghz")
    if standard_deviation < 0:
        raise ValueError("frequency_std_ghz must be non-negative")
    anharmonicity = _frequency_value(anharmonicity_ghz, "anharmonicity_ghz")
    if anharmonicity >= 0:
        raise ValueError("anharmonicity_ghz must be negative")
    if type(seed) is not int:
        raise TypeError("seed must be an exact non-Boolean integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    if abs(centers[0] - centers[1]) <= 6 * standard_deviation:
        warnings.warn(
            "The nominal three-standard-deviation frequency-group intervals "
            "touch or overlap.",
            UserWarning,
            stacklevel=2,
        )

    model, frequencies, edges = _model_document(
        rows=rows,
        columns=columns,
        centers=centers,
        standard_deviation=standard_deviation,
        anharmonicity=anharmonicity,
        seed=seed,
    )
    calibration = _calibration_document(
        edges=edges,
        frequencies=frequencies,
        anharmonicity=anharmonicity,
    )

    return model, calibration


__all__ = ["generate_transmon_grid_documents"]
