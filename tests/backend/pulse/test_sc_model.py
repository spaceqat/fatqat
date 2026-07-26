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
    assert first.bind_frame(first.frame("q0")) == 0
    assert first.bind_coupling(first.coupling("q1", "q0")) == 0
    with pytest.raises(BackendValidationError, match="foreign"):
        first.bind_resource(second.resource("q0"))
    with pytest.raises(BackendValidationError, match="unknown model subsystem"):
        first.resource("missing")


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
