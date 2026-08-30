"""Superconducting transmon physics-model value tests."""

from copy import deepcopy
import inspect

import pytest

from fatqat.emulator._core.target import _ControlAddress, _FrameAddress
from fatqat.emulator.superconducting.model import TransmonModel
from fatqat.errors import BackendValidationError


def test_model_is_direct_frozen_slotted_semantic_value(model_document):
    pristine = deepcopy(model_document)
    model = TransmonModel.from_document(model_document)
    model_document["parameters"]["subsystems"]["q0"]["frequency"] = 99
    assert tuple(inspect.signature(TransmonModel.from_document).parameters) == (
        "document",
    )
    assert model == TransmonModel.from_document(pristine)
    assert isinstance(model == TransmonModel.from_document(pristine), bool)
    changed_identity = deepcopy(pristine)
    changed_identity["model"]["revision"] = "different"
    assert model != TransmonModel.from_document(changed_identity)
    assert model.subsystem_ids == ("q0", "q1")
    assert model.basis_order == ("0", "1", "2")
    assert model.time_unit == "ns"
    with pytest.raises((AttributeError, TypeError)):
        model.subsystem_ids = ("changed",)
    with pytest.raises(TypeError):
        hash(model)
    for removed in (
        "format",
        "identity",
        "subsystems",
        "couplings",
        "kind",
        "local_dimension",
        "physical_dimension",
        "annihilation",
        "number",
        "frequency_unit",
        "anharmonicity_unit",
        "control_unit",
    ):
        assert not hasattr(model, removed)


def test_direct_construction_is_removed(model_document):
    with pytest.raises(TypeError, match="from_document"):
        TransmonModel()
    with pytest.raises(TypeError, match="from_document"):
        TransmonModel(model_document)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "format"),
        (("format",), "id"),
        (("model",), "id"),
        (("system",), "subsystem_type"),
        (("system",), "subsystems"),
        (("system",), "control_edges"),
        (("system", "control_edges", 0), "id"),
        (("units",), "frequency"),
        (("parameters",), "subsystems"),
        (("parameters", "subsystems", "q0"), "frequency"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_model_requires_exact_schema(model_document, path, key, mutation):
    cursor = model_document
    for part in path:
        cursor = cursor[part]
    if mutation == "missing":
        del cursor[key]
    else:
        cursor["unexpected"] = None
    with pytest.raises(BackendValidationError):
        TransmonModel.from_document(model_document)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["format"].update(id="foreign"),
        lambda d: d["format"].update(version=True),
        lambda d: d["system"].update(subsystem_type="fluxonium"),
        lambda d: d["system"].update(subsystems=[]),
        lambda d: d["system"].update(subsystems=["q0", "q0"]),
        lambda d: d["system"].update(control_edges={}),
        lambda d: d["system"]["control_edges"][0].update(subsystems=["q0"]),
        lambda d: d["system"]["control_edges"].append(
            {"id": "e1", "subsystems": ["q1", "q0"]}
        ),
        lambda d: d["system"]["control_edges"][0].update(subsystems=["q0", "q0"]),
        lambda d: d["units"].update(frequency="MHz"),
        lambda d: d["parameters"]["subsystems"]["q0"].update(frequency=0),
        lambda d: d["parameters"]["subsystems"]["q0"].update(anharmonicity=0),
        lambda d: d.update(callback=lambda: None),
    ],
)
def test_model_rejects_invalid_topology_and_parameters(model_document, mutate):
    mutate(model_document)
    with pytest.raises(BackendValidationError):
        TransmonModel.from_document(model_document)


def test_undirected_edge_order_has_no_semantic_effect(model_document):
    reversed_edge = deepcopy(model_document)
    reversed_edge["system"]["control_edges"][0]["subsystems"].reverse()
    assert TransmonModel.from_document(model_document) == TransmonModel.from_document(
        reversed_edge
    )


def test_model_factories_return_portable_structural_addresses(model_document):
    first = TransmonModel.from_document(model_document)
    second = TransmonModel.from_document(deepcopy(model_document))
    assert isinstance(first.control.drive("q0"), _ControlAddress)
    assert isinstance(first.control.detuning("q0"), _ControlAddress)
    assert isinstance(first.control.exchange("q0", "q1"), _ControlAddress)
    assert isinstance(first.frame("q0"), _FrameAddress)
    assert first.control.drive("q0") == second.control.drive("q0")
    assert first.control.exchange("q1", "q0") == second.control.exchange("q0", "q1")
    assert first.frame("q0") == second.frame("q0")
    with pytest.raises(BackendValidationError):
        first.control.exchange("q0", "q0")
    for removed in (
        "resource",
        "coupling",
        "bind_resource",
        "bind_control",
        "bind_frame",
        "_bind_gate_operands",
        "_bind_gate_control",
        "_bind_gate_frame",
        "validate_pulse_controls",
    ):
        assert not hasattr(first, removed)


def test_model_copies_stored_topology_containers(model_document):
    pristine = deepcopy(model_document)
    model = TransmonModel.from_document(model_document)
    model_document["system"]["subsystems"][0] = "changed"
    model_document["system"]["control_edges"][0]["subsystems"].reverse()
    model_document["system"]["control_edges"].clear()
    assert model.subsystem_ids == ("q0", "q1")
    assert model == TransmonModel.from_document(pristine)


def test_arbitrary_connectivity_is_valid(model_document):
    model_document["system"]["subsystems"].append("q2")
    model_document["parameters"]["subsystems"]["q2"] = {
        "frequency": 5.3,
        "anharmonicity": -0.2,
    }
    assert TransmonModel.from_document(model_document).subsystem_ids == (
        "q0",
        "q1",
        "q2",
    )
