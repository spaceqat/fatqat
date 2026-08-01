"""Shared fixtures for the emulator suite.

A loaded model and calibration are immutable, so one session-scoped build is
safe to share and removes the loader boilerplate that was duplicated across
eight test modules.

Two escape hatches exist on purpose:

* `model_document` / `calibration_document` are function-scoped fresh copies,
  for tests that mutate a document before loading it.
* `build_model_and_calibration` is a factory, for tests that need a second,
  *distinct* model instance. Handle identity is tied to the minting instance,
  so foreign-handle tests must not reuse the shared `model` fixture.
"""

import json
from pathlib import Path

import pytest

from fatqat.emulator.backend import PulseBackend
from fatqat.emulator.superconducting import (
    load_calibration_spec,
    load_physics_model,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture(name="model_document")
def model_document_fixture() -> dict:
    """A fresh, mutable copy of the model document."""
    return _read("sc_transmon_exchange.json")


@pytest.fixture(name="calibration_document")
def calibration_document_fixture() -> dict:
    """A fresh, mutable copy of the calibration document."""
    return _read("sc_transmon_exchange_calibration.json")


@pytest.fixture(name="model", scope="session")
def model_fixture():
    """The shared immutable physics model."""
    return load_physics_model(_read("sc_transmon_exchange.json"))


@pytest.fixture(name="calibration", scope="session")
def calibration_fixture(model):
    """The shared calibration, identity-bound to `model`."""
    return load_calibration_spec(_read("sc_transmon_exchange_calibration.json"), model)


@pytest.fixture(name="build_model_and_calibration")
def build_model_and_calibration_fixture():
    """Return a factory building an independent model/calibration pair.

    Each call mints fresh handles, so two builds are mutually foreign.
    """

    def build():
        built = load_physics_model(_read("sc_transmon_exchange.json"))
        return built, load_calibration_spec(
            _read("sc_transmon_exchange_calibration.json"), built
        )

    return build


@pytest.fixture(name="backend")
def backend_fixture(model, calibration) -> PulseBackend:
    """A noise-free backend on the shared model/calibration pair.

    Function-scoped: `PulseBackend` owns a mutable noise model and private
    copies of the implementation maps, and several tests register noise on it.
    """
    return PulseBackend(model, calibration)
