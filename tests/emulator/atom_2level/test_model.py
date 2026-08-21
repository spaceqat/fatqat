"""Strict two-level atom document-construction and channel contracts."""

from copy import deepcopy
from dataclasses import fields
import inspect
import json
from math import inf, nan
from pathlib import Path

import pytest

from fatqat._pulse_values import ControlChannel
from fatqat.emulator._core.model_document import FormatIdentity, ModelIdentity
from fatqat.emulator._core.target import _ControlAddress
from fatqat.emulator.atom_2level.model import Atom2LevelModel, _MODEL_PARSERS
from fatqat.errors import BackendValidationError

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="document")
def document_fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_from_document_is_geometry_free_immutable_and_unit_explicit(document):
    assert tuple(inspect.signature(Atom2LevelModel.from_document).parameters) == (
        "document",
    )
    source = deepcopy(document)
    model = Atom2LevelModel.from_document(document)
    document["system"]["basis"]["g"] = "changed"

    assert model == Atom2LevelModel.from_document(source)
    assert isinstance(model == Atom2LevelModel.from_document(source), bool)
    assert model.format == FormatIdentity("atom.rb87_rydberg_2level", 1)
    assert model.kind == "atom.rydberg_2level"
    assert model.identity == ModelIdentity("rb87-70s-analog-reference", "2026-08-07")
    assert model.species == "Rb87"
    assert model.ground_state == "5S1/2,F=2,mF=2"
    assert model.basis_order == ("g", "r")
    assert model.local_dimension == 2
    assert model.interaction_law == "C6/R^6"
    assert model.c6_angular_per_us_um6 == 1.0
    assert (
        model.distance_unit,
        model.time_unit,
        model.angular_frequency_unit,
        model.c6_unit,
    ) == ("um", "us", "rad/us", "rad/us*um^6")
    assert not hasattr(model, "units")
    assert not hasattr(model, "system")
    assert not hasattr(model, "builder")
    assert not hasattr(model, "channel_limits")
    with pytest.raises(TypeError):
        hash(model)
    with pytest.raises(AttributeError):
        model.identity = object()
    with pytest.raises(AttributeError):
        del model.identity


def test_direct_construction_is_removed(document):
    with pytest.raises(TypeError, match="from_document"):
        Atom2LevelModel()
    with pytest.raises(TypeError, match="from_document"):
        Atom2LevelModel(document)


@pytest.mark.parametrize("c6", [-2.0, 0.0, 2.0])
def test_constructor_accepts_finite_signed_or_zero_c6(document, c6):
    document["parameters"]["c6"] = c6
    assert Atom2LevelModel.from_document(document).c6_angular_per_us_um6 == c6


@pytest.mark.parametrize(
    ("path", "known_key"),
    [
        ((), "format"),
        (("format",), "id"),
        (("model",), "id"),
        (("system",), "species"),
        (("system", "basis"), "g"),
        (("system", "transitions", "rydberg"), "from"),
        (("units",), "distance"),
        (("parameters",), "c6"),
        (("parameters", "channel_limits"), "rydberg_global"),
        (("parameters", "channel_limits", "rydberg_global"), "max_amplitude"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_constructor_requires_exact_keys_at_every_level(
    document, path, known_key, mutation
):
    cursor = document
    for key in path:
        cursor = cursor[key]
    if mutation == "missing":
        del cursor[known_key]
    else:
        cursor["unexpected"] = None
    with pytest.raises(BackendValidationError):
        Atom2LevelModel.from_document(document)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["format"].update(id="foreign"),
        lambda d: d["format"].update(version=True),
        lambda d: d["system"].update(species="Cs133"),
        lambda d: d["system"]["basis"].update(g=""),
        lambda d: d["system"]["basis"].update(r=d["system"]["basis"]["g"]),
        lambda d: d["system"]["transitions"]["rydberg"].update(**{"from": "r"}),
        lambda d: d["units"].update(time="ns"),
        lambda d: d["parameters"].update(c6=True),
        lambda d: d["parameters"].update(c6=nan),
        lambda d: d.update(callback=lambda: None),
    ],
)
def test_constructor_rejects_invalid_fixed_model_facts(document, mutate):
    mutate(document)
    with pytest.raises(BackendValidationError):
        Atom2LevelModel.from_document(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_amplitude", True),
        ("max_amplitude", -1),
        ("max_amplitude", inf),
        ("min_detuning", nan),
        ("max_detuning", -inf),
        ("min_duration", 0),
        ("max_duration", -1),
    ],
)
def test_constructor_rejects_invalid_optional_limits(document, field, value):
    document["parameters"]["channel_limits"]["rydberg_global"][field] = value
    with pytest.raises(BackendValidationError):
        Atom2LevelModel.from_document(document)


@pytest.mark.parametrize(
    "updates",
    [
        {"min_detuning": 2, "max_detuning": 1},
        {"min_duration": 2, "max_duration": 1},
    ],
)
def test_constructor_rejects_inverted_limit_pairs(document, updates):
    document["parameters"]["channel_limits"]["rydberg_global"].update(updates)
    with pytest.raises(BackendValidationError):
        Atom2LevelModel.from_document(document)


def test_control_selectors_have_exact_zero_argument_structural_contract(document):
    first = Atom2LevelModel.from_document(document)
    changed = deepcopy(document)
    changed["parameters"]["c6"] = -2.0
    second = Atom2LevelModel.from_document(changed)

    assert isinstance(first.control.drive(), ControlChannel)
    assert isinstance(first.control.detuning(), ControlChannel)
    assert first.control.drive() == second.control.drive()
    assert first.control.detuning() == second.control.detuning()
    assert first.control.drive() != first.control.detuning()
    assert type(first.control.drive()).__name__.startswith("_")
    with pytest.raises(TypeError):
        first.control.drive(0)


def test_removed_legacy_discovery_types_are_not_public(document):
    import fatqat.emulator.atom_2level as atom_2level

    model = Atom2LevelModel.from_document(document)
    assert not hasattr(atom_2level, "Channel" + "Description")
    assert not hasattr(atom_2level, "ControlComponent" + "Description")
    assert not hasattr(model, "describe_" + "channel")


def test_model_is_a_direct_frozen_slotted_value_with_exact_parser_table(document):
    model = Atom2LevelModel.from_document(document)
    assert tuple(_MODEL_PARSERS) == (FormatIdentity("atom.rb87_rydberg_2level", 1),)
    assert not next(
        field for field in fields(Atom2LevelModel) if field.name == "format"
    ).compare
    with pytest.raises(TypeError):
        _MODEL_PARSERS[FormatIdentity("other", 1)] = object()
    with pytest.raises(TypeError):
        del _MODEL_PARSERS[FormatIdentity("atom.rb87_rydberg_2level", 1)]
    assert hasattr(Atom2LevelModel, "__slots__")
    for removed in (
        "_normalized" + "_state",
        "_model" + "_key",
        "registry",
        "parser",
    ):
        assert not hasattr(model, removed)
    assert isinstance(model.control.drive(), _ControlAddress)
    assert isinstance(model.control.detuning(), _ControlAddress)
