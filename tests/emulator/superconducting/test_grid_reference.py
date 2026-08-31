"""Generated transmon-grid reference behavior tests."""

from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops


def _generate(**overrides):
    arguments = {
        "shape": (2, 2),
        "frequency_groups_ghz": (5.0, 5.2),
        "frequency_std_ghz": 0.01,
        "anharmonicity_ghz": -0.22,
        "seed": 0,
    }
    arguments.update(overrides)
    return fq.emulator.generate_transmon_grid_documents(**arguments)


def _revision_without_declared_revision(document, identity_key):
    material = deepcopy(document)
    del material[identity_key]["revision"]
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _frequencies(documents):
    model_document, _calibration_document = documents
    return model_document["parameters"]["subsystems"]


def test_generator_signature_and_document_ownership_are_explicit():
    function = fq.emulator.generate_transmon_grid_documents
    parameters = inspect.signature(function).parameters
    assert tuple(parameters) == (
        "shape",
        "frequency_groups_ghz",
        "frequency_std_ghz",
        "anharmonicity_ghz",
        "seed",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert parameters["shape"].default is inspect.Parameter.empty
    assert parameters["frequency_groups_ghz"].default is inspect.Parameter.empty
    assert parameters["frequency_std_ghz"].default == 0.010
    assert parameters["anharmonicity_ghz"].default == -0.22
    assert parameters["seed"].default == 0

    documents = _generate()
    repeated = _generate()
    assert isinstance(documents, tuple)
    assert len(documents) == 2
    assert all(isinstance(document, dict) for document in documents)
    assert documents == repeated
    assert documents != _generate(seed=1)
    assert documents[0] is not repeated[0]
    assert documents[1] is not repeated[1]

    documents[0]["parameters"]["subsystems"]["q0"]["frequency"] = 99.0
    documents[1]["recipes"]["rx_ry"]["duration"] = 99.0
    assert repeated[0]["parameters"]["subsystems"]["q0"]["frequency"] != 99.0
    assert repeated[1]["recipes"]["rx_ry"]["duration"] != 99.0


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"shape": [2, 2]}, TypeError),
        ({"shape": (2,)}, ValueError),
        ({"shape": (True, 2)}, TypeError),
        ({"shape": (2.0, 2)}, TypeError),
        ({"shape": (0, 2)}, ValueError),
        ({"shape": (1, 1)}, ValueError),
        ({"frequency_groups_ghz": [5.0, 5.2]}, TypeError),
        ({"frequency_groups_ghz": (5.0,)}, ValueError),
        ({"frequency_groups_ghz": (True, 5.2)}, TypeError),
        ({"frequency_groups_ghz": (5.0, False)}, TypeError),
        ({"frequency_groups_ghz": (0.0, 5.2)}, ValueError),
        ({"frequency_groups_ghz": (5.0, 5.0)}, ValueError),
        ({"frequency_groups_ghz": (math.nan, 5.2)}, ValueError),
        ({"frequency_groups_ghz": (5.0, math.inf)}, ValueError),
        ({"frequency_std_ghz": True}, TypeError),
        ({"frequency_std_ghz": -0.1}, ValueError),
        ({"frequency_std_ghz": math.inf}, ValueError),
        ({"anharmonicity_ghz": False}, TypeError),
        ({"anharmonicity_ghz": 0.0}, ValueError),
        ({"anharmonicity_ghz": math.nan}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": -1}, ValueError),
    ],
)
def test_generator_rejects_invalid_inputs(overrides, error):
    with pytest.raises(error):
        _generate(**overrides)


def test_numeric_inputs_normalize_and_realized_overflow_names_its_site():
    integral = _generate(
        frequency_groups_ghz=(5, 6),
        frequency_std_ghz=0,
        anharmonicity_ghz=-1,
    )
    floating = _generate(
        frequency_groups_ghz=(5.0, 6.0),
        frequency_std_ghz=0.0,
        anharmonicity_ghz=-1.0,
    )
    assert integral == floating
    with pytest.warns(UserWarning, match="three-standard-deviation"):
        with pytest.raises(ValueError, match=r"site q0.*finite and positive"):
            _generate(
                shape=(1, 2),
                frequency_groups_ghz=(1e308, 9e307),
                frequency_std_ghz=1e308,
                seed=7,
            )


def test_zero_jitter_uses_row_column_checkerboard_parameters():
    reference = _generate(frequency_std_ghz=0, anharmonicity_ghz=-0.3)
    parameters = _frequencies(reference)
    assert [parameters[f"q{i}"]["frequency"] for i in range(4)] == [
        5.0,
        5.2,
        5.2,
        5.0,
    ]
    assert {item["anharmonicity"] for item in parameters.values()} == {-0.3}


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (
            (2, 3),
            [
                ["q0", "q1"],
                ["q0", "q3"],
                ["q1", "q2"],
                ["q1", "q4"],
                ["q2", "q5"],
                ["q3", "q4"],
                ["q4", "q5"],
            ],
        ),
        ((1, 4), [["q0", "q1"], ["q1", "q2"], ["q2", "q3"]]),
        ((4, 1), [["q0", "q1"], ["q1", "q2"], ["q2", "q3"]]),
    ],
)
def test_edges_follow_right_then_down_traversal(shape, expected):
    document, _calibration = _generate(shape=shape, frequency_std_ghz=0)
    assert document["system"]["subsystems"] == [
        f"q{index}" for index in range(shape[0] * shape[1])
    ]
    assert document["system"]["control_edges"] == [
        {"id": f"e{index}", "subsystems": edge} for index, edge in enumerate(expected)
    ]


def test_edge_canonicalization_uses_string_order_not_numeric_suffixes():
    document, _calibration = _generate(shape=(1, 11), frequency_std_ghz=0)
    edges = document["system"]["control_edges"]
    assert edges[-1] == {"id": "e9", "subsystems": ["q10", "q9"]}


def test_site_keyed_rng_is_repeatable_extensible_and_matches_v1_goldens():
    base = _generate(shape=(2, 3), seed=0)
    repeated = _generate(shape=(2, 3), seed=0)
    changed_seed = _generate(shape=(2, 3), seed=1)
    extended = _generate(shape=(3, 3), seed=0)
    assert base == repeated
    assert base[0] != changed_seed[0]

    base_parameters = _frequencies(base)
    extended_parameters = _frequencies(extended)
    assert base_parameters == {
        label: extended_parameters[label] for label in base_parameters
    }

    expected_draws = (
        -1.0657422531523777,
        -1.212682998519296,
        -1.726370902280585,
        -0.9714428283269375,
    )
    parameters = _frequencies(_generate(seed=0))
    centers = (5.0, 5.2, 5.2, 5.0)
    for index, (center, draw) in enumerate(zip(centers, expected_draws)):
        assert parameters[f"q{index}"]["frequency"] == pytest.approx(
            center + 0.01 * draw
        )

    doubled = _frequencies(_generate(seed=0, frequency_std_ghz=0.02))
    for index, center in enumerate(centers):
        label = f"q{index}"
        assert (doubled[label]["frequency"] - center) / 0.02 == pytest.approx(
            (parameters[label]["frequency"] - center) / 0.01
        )


def test_generation_leaves_ambient_numpy_rng_untouched_and_zero_jitter_skips_seed():
    before = np.random.get_state()
    _generate(seed=123)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert _generate(frequency_std_ghz=0, seed=0) == _generate(
        frequency_std_ghz=0, seed=10**100
    )


def test_nominal_overlap_warning_includes_the_three_sigma_boundary():
    with warnings.catch_warnings(record=True) as boundary:
        warnings.simplefilter("always")
        _generate(
            shape=(1, 2),
            frequency_groups_ghz=(10.0, 16.0),
            frequency_std_ghz=1.0,
        )
    assert [item.category for item in boundary] == [UserWarning]
    assert "three-standard-deviation" in str(boundary[0].message)

    with warnings.catch_warnings(record=True) as separated:
        warnings.simplefilter("always")
        _generate(
            shape=(1, 2),
            frequency_groups_ghz=(10.0, 16.0000000001),
            frequency_std_ghz=1.0,
        )
    assert not separated


@pytest.mark.parametrize(
    ("arguments", "expected_fragments"),
    [
        (
            {
                "shape": (2, 3),
                "frequency_groups_ghz": (10.0, 10.72714200189354),
                "frequency_std_ghz": 1.0,
                "seed": 1,
            },
            ("touch or overlap", "0 nearest-neighbor", "none"),
        ),
        (
            {
                "shape": (2, 2),
                "frequency_groups_ghz": (10.0, 10.1),
                "frequency_std_ghz": 1.0,
                "seed": 0,
            },
            ("do not overlap", "4 nearest-neighbor", "(q0, q1)"),
        ),
    ],
)
def test_realized_overlap_and_reversal_warnings_are_aggregated(
    arguments, expected_fragments
):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        reference = _generate(**arguments)
    assert len(seen) == 2
    assert all(item.category is UserWarning for item in seen)
    assert all(
        Path(item.filename).resolve() == Path(__file__).resolve() for item in seen
    )
    realized_message = str(seen[1].message)
    assert all(fragment in realized_message for fragment in expected_fragments)

    if arguments["seed"] == 0:
        assert _frequencies(reference)["q0"]["frequency"] == pytest.approx(
            10.0 - 1.0657422531523777
        )


def test_realized_warning_caps_reported_edge_labels():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _generate(
            shape=(10, 10),
            frequency_groups_ghz=(10.0, 10.1),
            frequency_std_ghz=1.0,
            seed=0,
        )
    realized_message = str(seen[1].message)
    assert "... and " in realized_message
    assert len(realized_message) < 1_000


def test_exact_realized_tie_warns_and_selects_the_canonical_first_endpoint():
    lower = 1.0
    upper = math.nextafter(lower, math.inf)
    standard_deviation = (upper - lower) / 2
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        reference = _generate(
            shape=(1, 2),
            frequency_groups_ghz=(lower, upper),
            frequency_std_ghz=standard_deviation,
            seed=2,
        )
    parameters = _frequencies(reference)
    assert parameters["q0"]["frequency"] == parameters["q1"]["frequency"]
    _model, calibration = reference
    entry = calibration["recipes"]["cz"]["edges"][0]
    assert entry["canonical_edge"] == ["q0", "q1"]
    assert entry["recipe"]["detuned_subsystem"] == "q0"
    assert any("1 nearest-neighbor" in str(item.message) for item in seen)


def test_nonpositive_realized_frequency_fails_with_site_context():
    with pytest.raises(ValueError, match=r"site q0.*positive"):
        _generate(
            shape=(1, 2),
            frequency_groups_ghz=(1.0, 10.0),
            frequency_std_ghz=1.0,
            seed=0,
        )


def test_documents_have_content_revisions_and_portable_analytic_recipes():
    reference = _generate(shape=(2, 3), anharmonicity_ghz=-0.25)
    model, calibration = reference
    revision_pattern = re.compile(r"sha256:[0-9a-f]{64}\Z")
    assert model["model"]["id"] == "synthetic-transmon-grid-reference"
    assert calibration["calibration"]["id"] == (
        "fatqat_generated_transmon_grid_reference"
    )
    assert revision_pattern.fullmatch(model["model"]["revision"])
    assert revision_pattern.fullmatch(calibration["calibration"]["revision"])
    assert model["model"]["revision"] == _revision_without_declared_revision(
        model, "model"
    )
    assert calibration["calibration"][
        "revision"
    ] == _revision_without_declared_revision(calibration, "calibration")

    model_edges = [edge["subsystems"] for edge in model["system"]["control_edges"]]
    cz_entries = calibration["recipes"]["cz"]["edges"]
    assert [entry["canonical_edge"] for entry in cz_entries] == model_edges
    frequencies = model["parameters"]["subsystems"]
    for entry in cz_entries:
        first, second = entry["canonical_edge"]
        expected = (
            first
            if frequencies[first]["frequency"] >= frequencies[second]["frequency"]
            else second
        )
        assert entry["recipe"] == {
            "detuned_subsystem": expected,
            "duration": 60.0,
            "ramp_duration": 3.0,
            "park_detuning_ghz": 0.25,
            "branch_tolerance_ghz": 1e-12,
        }
    assert calibration["recipes"]["rx_ry"] == {
        "duration": 20.0,
        "drag_coefficient": 1.0,
    }
    assert calibration["recipes"]["iswap"] == {"duration": 40.0}
    assert calibration["provenance"] == {
        "kind": "generated_reference_recipe",
        "generator_version": 1,
        "numerically_calibrated": False,
    }
    assert set(calibration) == {
        "format",
        "calibration",
        "provenance",
        "units",
        "recipes",
    }
    encoded = json.dumps(calibration).lower()
    for forbidden in ("model", "seed", "fidelity", "leakage", "qualification"):
        assert forbidden not in encoded

    horizontal = _generate(
        shape=(1, 3), frequency_groups_ghz=(5, 6), frequency_std_ghz=0
    )
    vertical = _generate(shape=(3, 1), frequency_groups_ghz=(5, 6), frequency_std_ghz=0)
    assert horizontal == vertical


def test_documents_round_trip_and_build_an_explicit_multiedge_emulator():
    model_document, calibration_document = _generate(shape=(2, 2))
    model = fq.emulator.TransmonModel.from_document(
        json.loads(json.dumps(model_document))
    )
    calibration = fq.emulator.TransmonCalibration(
        json.loads(json.dumps(calibration_document))
    )
    implementations = fq.emulator.default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    backend = fq.emulator.TransmonEmulator(
        model, gate_implementation_map=implementations
    )
    assert backend.model is model

    first, second = model_document["system"]["control_edges"][0]["subsystems"]
    forward_rule = implementations.implementation_for(
        ops.CZ, device_operands=(first, second)
    )
    reverse_rule = implementations.implementation_for(
        ops.CZ, device_operands=(second, first)
    )
    forward = forward_rule(ops.CZ, device_operands=(first, second))
    reverse = reverse_rule(ops.CZ, device_operands=(second, first))
    assert forward == reverse


def test_document_generation_does_not_enter_realization_or_emulation(monkeypatch):
    import fatqat.emulator.superconducting as superconducting

    def unexpected(*_args, **_kwargs):
        raise AssertionError("generation entered a numerical or realization path")

    monkeypatch.setattr(
        superconducting, "default_transmon_gate_implementation_map", unexpected
    )
    monkeypatch.setattr(superconducting, "TransmonEmulator", unexpected)
    model_document, calibration_document = (
        superconducting.generate_transmon_grid_documents(
            shape=(1, 2), frequency_groups_ghz=(5.0, 5.2)
        )
    )
    assert model_document["system"]["subsystems"] == ["q0", "q1"]
    assert calibration_document["recipes"]["cz"]["edges"]
