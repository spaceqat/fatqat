"""Superconducting transmon physics-model value tests."""

from copy import deepcopy
import inspect

import numpy as np
import pytest

from fatqat.emulator._core.model_document import FormatIdentity
from fatqat.emulator._core.target import _ControlAddress, _FrameAddress
from fatqat.emulator.superconducting.model import TransmonModel, _MODEL_PARSERS
from fatqat.errors import BackendValidationError


def test_model_is_direct_frozen_slotted_semantic_value(model_document):
    pristine = deepcopy(model_document)
    model = TransmonModel(model_document)
    model_document["parameters"]["subsystems"]["q0"]["frequency"] = 99
    assert tuple(inspect.signature(TransmonModel).parameters) == ("document",)
    assert model == TransmonModel(pristine)
    assert isinstance(model == TransmonModel(pristine), bool)
    assert model.subsystem_ids == ("q0", "q1")
    assert tuple(item.frequency_ghz for item in model.subsystems) == (5.1, 5.22)
    assert model.couplings[0].subsystem_ids == ("q0", "q1")
    assert np.array_equal(
        model.annihilation,
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2)], [0.0, 0.0, 0.0]]),
    )
    assert np.array_equal(model.number, np.diag([0.0, 1.0, 2.0]))
    assert not model.annihilation.flags.writeable
    assert (model.frequency_unit, model.anharmonicity_unit, model.time_unit) == (
        "GHz",
        "GHz",
        "ns",
    )
    with pytest.raises(TypeError):
        hash(model)
    with pytest.raises(AttributeError):
        model.identity = object()
    for removed in (
        "_normalized" + "_state",
        "_model" + "_key",
        "registry",
        "parser",
    ):
        assert not hasattr(model, removed)


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
        TransmonModel(model_document)


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
        TransmonModel(model_document)


def test_undirected_edge_order_has_no_semantic_effect(model_document):
    reversed_edge = deepcopy(model_document)
    reversed_edge["system"]["control_edges"][0]["subsystems"].reverse()
    assert TransmonModel(model_document) == TransmonModel(reversed_edge)


def test_model_factories_return_portable_structural_addresses(model_document):
    first = TransmonModel(model_document)
    second = TransmonModel(deepcopy(model_document))
    assert isinstance(first.drive_control("q0"), _ControlAddress)
    assert isinstance(first.detuning_control("q0"), _ControlAddress)
    assert isinstance(first.exchange_control("q0", "q1"), _ControlAddress)
    assert isinstance(first.frame("q0"), _FrameAddress)
    assert first.drive_control("q0") == second.drive_control("q0")
    assert first.exchange_control("q1", "q0") == second.exchange_control("q0", "q1")
    assert first.frame("q0") == second.frame("q0")
    with pytest.raises(BackendValidationError):
        first.exchange_control("q0", "q0")
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
    model = TransmonModel(model_document)
    model_document["system"]["subsystems"][0] = "changed"
    model_document["system"]["control_edges"][0]["subsystems"].reverse()
    model_document["system"]["control_edges"].clear()
    assert model.subsystem_ids == ("q0", "q1")
    assert model.couplings[0].subsystem_ids == ("q0", "q1")


def test_arbitrary_connectivity_is_valid(model_document):
    model_document["system"]["subsystems"].append("q2")
    model_document["parameters"]["subsystems"]["q2"] = {
        "frequency": 5.3,
        "anharmonicity": -0.2,
    }
    assert TransmonModel(model_document).subsystem_ids == ("q0", "q1", "q2")


def test_model_uses_one_immutable_exact_format_table():
    assert tuple(_MODEL_PARSERS) == (FormatIdentity("sc.transmon_exchange", 1),)
    with pytest.raises(TypeError):
        _MODEL_PARSERS[FormatIdentity("other", 1)] = object()
    with pytest.raises(TypeError):
        del _MODEL_PARSERS[FormatIdentity("sc.transmon_exchange", 1)]
