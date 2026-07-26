"""Built-model local facts and opaque-handle checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from fatqat.backends.pulse.superconducting import load_physics_model
from fatqat.errors import BackendValidationError

_FIXTURES = Path(__file__).parent / "fixtures"


def _document():
    return json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())


def test_model_contains_only_local_qutrit_facts_and_expected_ladder_action():
    model = load_physics_model(_document())

    assert model.physical_dimension == 3
    assert model.time_unit == "ns"
    assert model.annihilation.shape == (3, 3)
    assert np.allclose(model.annihilation @ [0, 1, 0], [1, 0, 0])
    assert np.allclose(model.annihilation @ [0, 0, 1], [0, np.sqrt(2), 0])
    assert np.allclose(model.creation, model.annihilation.conj().T)
    assert np.allclose(model.number @ [0, 0, 1], [0, 0, 2])
    assert not model.annihilation.flags.writeable
    assert not hasattr(model, "qobj")
    assert not hasattr(model, "solver_cache")


def test_same_model_handles_bind_and_foreign_or_unknown_handles_fail():
    first = load_physics_model(_document())
    second = load_physics_model(_document())

    assert first.bind_resource(first.resource("q0")) == 0
    assert first.bind_control(first.drive_control("q1")) == 1
    assert first.bind_control(first.exchange_control("q1", "q0")) == 0
    assert first.bind_frame(first.frame("q0")) == 0
    assert first.bind_coupling(first.coupling("q1", "q0")) == 0
    with pytest.raises(BackendValidationError, match="foreign"):
        first.bind_resource(second.resource("q0"))
    with pytest.raises(BackendValidationError, match="unknown model subsystem"):
        first.resource("missing")


def test_handle_identity_ties_equality_to_minting_provenance_not_just_key():
    first = load_physics_model(_document())
    second = load_physics_model(_document())

    # Same instance: repeated lookups return the identical, equal handle.
    assert first.resource("q0") == first.resource("q0")
    assert first.drive_control("q0") == first.drive_control("q0")
    assert first.detuning_control("q0") == first.detuning_control("q0")
    assert first.exchange_control("q0", "q1") == first.exchange_control("q0", "q1")
    assert first.frame("q0") == first.frame("q0")
    assert first.coupling("q0", "q1") == first.coupling("q0", "q1")

    # Different instances built from the identical persisted model key mint
    # handles that share model_key/ordinal but must stay unequal: `bind_*`
    # rejects `second`'s handles against `first`, and equality must agree.
    assert first.resource("q0").model_key == second.resource("q0").model_key
    assert first.resource("q0").ordinal == second.resource("q0").ordinal
    assert first.resource("q0") != second.resource("q0")
    assert hash(first.resource("q0")) != hash(second.resource("q0"))
    assert first.drive_control("q0") != second.drive_control("q0")
    assert first.exchange_control("q0", "q1") != second.exchange_control("q0", "q1")
    assert first.frame("q0") != second.frame("q0")
    assert first.coupling("q0", "q1") != second.coupling("q0", "q1")
    with pytest.raises(BackendValidationError, match="foreign"):
        first.bind_resource(second.resource("q0"))


def test_exchange_control_is_coupling_sized_and_distinct_from_pair_resource():
    model = load_physics_model(_document())

    assert len(model.couplings) == 1
    exchange = model.exchange_control("q0", "q1")
    assert exchange.kind == "exchange"
    assert exchange.ordinal == 0
    assert exchange != model.coupling("q0", "q1")
    # Edge lookup is undirected, like `coupling(...)`.
    assert exchange == model.exchange_control("q1", "q0")


def test_arbitrary_connectivity_allows_disconnected_and_single_transmon_models():
    disconnected = _document()
    disconnected["parameters"]["couplings"] = []
    model = load_physics_model(disconnected)
    assert model.couplings == ()
    with pytest.raises(BackendValidationError, match="no declared coupling"):
        model.coupling("q0", "q1")

    single = _document()
    single["parameters"]["subsystems"] = single["parameters"]["subsystems"][:1]
    single["parameters"]["couplings"] = []
    assert load_physics_model(single).subsystem_ids == ("q0",)
