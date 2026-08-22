"""Public reference-model catalog contract."""

from copy import deepcopy

import pytest

import fatqat as fq

_EXPECTED = {
    "atom2level.reference": (
        fq.emulator.Atom2LevelModel,
        "atom.rb87_rydberg_2level",
        "synthetic-atom2level-reference",
        "2026-08-22",
        {
            "distance": "um",
            "time": "us",
            "angular_frequency": "rad/us",
            "c6": "rad/us*um^6",
        },
    ),
    "atom3level.reference": (
        fq.emulator.Atom3LevelModel,
        "atom.rb87_rydberg_3level",
        "rb87-53s-reference",
        "2026-08-05",
        {
            "mass": "u",
            "distance": "um",
            "time": "us",
            "angular_frequency": "rad/us",
            "c6": "rad/us*um^6",
        },
    ),
    "transmon.reference": (
        fq.emulator.TransmonModel,
        "sc.transmon_exchange",
        "test-sc-2q",
        "2026-07-26",
        {"frequency": "GHz", "anharmonicity": "GHz"},
    ),
}


def test_available_model_documents_is_exact_and_deterministic():
    assert fq.emulator.available_model_documents() == tuple(_EXPECTED)


@pytest.mark.parametrize("invalid", [None, 1, True, object()])
def test_load_model_document_rejects_non_string_name(invalid):
    with pytest.raises(TypeError, match="name must be a string"):
        fq.emulator.load_model_document(invalid)


def test_load_model_document_rejects_unknown_name_without_resource_details():
    with pytest.raises(KeyError, match="unknown model document") as captured:
        fq.emulator.load_model_document("missing.reference")

    message = str(captured.value)
    assert "_model_documents" not in message
    assert ".json" not in message


def test_load_model_document_returns_independent_mutable_graphs():
    first = fq.emulator.load_model_document("atom3level.reference")
    original = deepcopy(first)
    first["parameters"]["mass"] = 1.0
    first["provenance"]["sources"].append("changed")

    second = fq.emulator.load_model_document("atom3level.reference")
    assert second == original


@pytest.mark.parametrize(
    ("name", "expected"),
    _EXPECTED.items(),
)
def test_reference_document_has_expected_payload_and_constructs(name, expected):
    model_type, format_id, model_id, revision, units = expected
    document = fq.emulator.load_model_document(name)

    assert isinstance(document, dict)
    assert document["format"] == {"id": format_id, "version": 1}
    assert document["model"] == {"id": model_id, "revision": revision}
    assert document["units"] == units
    assert isinstance(document["parameters"], dict) and document["parameters"]
    assert set(document["provenance"]) == {"description", "sources"}
    assert document["provenance"]["description"]
    assert isinstance(document["provenance"]["sources"], list)
    assert isinstance(model_type.from_document(document), model_type)


@pytest.mark.parametrize(
    ("name", "parameters", "source_marker"),
    [
        (
            "atom2level.reference",
            {
                "c6": 1.0,
                "channel_limits": {
                    "rydberg_global": {
                        "max_amplitude": None,
                        "min_detuning": None,
                        "max_detuning": None,
                        "min_duration": None,
                        "max_duration": None,
                    }
                },
            },
            None,
        ),
        (
            "atom3level.reference",
            {"mass": 86.9091805, "c6": 180955.73684677208},
            "10.1038/s41586-023-06481-y",
        ),
        (
            "transmon.reference",
            {
                "subsystems": {
                    "q0": {"frequency": 5.1, "anharmonicity": -0.22},
                    "q1": {"frequency": 5.22, "anharmonicity": -0.24},
                }
            },
            None,
        ),
    ],
)
def test_reference_parameters_and_provenance_are_pinned(
    name, parameters, source_marker
):
    document = fq.emulator.load_model_document(name)
    provenance = document["provenance"]

    assert document["parameters"] == parameters
    if source_marker is None:
        assert provenance["sources"] == []
        assert "synthetic" in provenance["description"].lower()
    else:
        assert source_marker in provenance["sources"][0]


def test_private_catalog_resources_are_not_public_exports():
    assert "_model_documents" not in fq.emulator.__all__
    assert "_DOCUMENT_RESOURCES" not in fq.emulator.__all__
