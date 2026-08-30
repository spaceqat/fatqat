"""Shared fixtures for the emulator suite.

A loaded model and calibration are immutable, so one session-scoped build is
safe to share and removes the loader boilerplate that was duplicated across
eight test modules.

Two escape hatches exist on purpose:

* `model_document` / `calibration_document` are function-scoped fresh copies,
  for tests that mutate a document before loading it.
* `build_model_and_calibration` is a factory, for tests that need a second,
  *distinct* source model instance and need to prove structural addresses are
  portable across compatible targets.
"""

import json
from pathlib import Path

import pytest

from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator.superconducting import (
    TransmonCalibration,
    TransmonModel,
)
from fatqat.emulator.atom_3level import Atom3LevelCalibration, Atom3LevelModel

_FIXTURES = Path(__file__).parent / "fixtures"
_ATOM_3LEVEL_FIXTURES = Path(__file__).parent / "atom_3level" / "fixtures"


def _read(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _read_atom_3level(name: str) -> dict:
    return json.loads((_ATOM_3LEVEL_FIXTURES / name).read_text())


@pytest.fixture(name="model_document")
def model_document_fixture() -> dict:
    """A fresh, mutable copy of the model document."""
    return _read("sc_transmon_exchange.json")


@pytest.fixture(name="calibration_document")
def calibration_document_fixture() -> dict:
    """A fresh, mutable copy of the calibration document."""
    return _read("sc_transmon_exchange_calibration.json")


@pytest.fixture(name="atom_3level_model_document")
def atom_3level_model_document_fixture() -> dict:
    """A fresh three-level-atom model document for private atom tests."""
    return _read_atom_3level("atom_3level_rb87_53s.json")


@pytest.fixture(name="atom_3level_calibration_document")
def atom_3level_calibration_document_fixture() -> dict:
    """A fresh three-level-atom calibration document for private atom tests."""
    return _read_atom_3level("atom_3level_rb87_53s_lukin_2023_calibration.json")


@pytest.fixture(name="atom_3level_model")
def atom_3level_model_fixture(atom_3level_model_document):
    return Atom3LevelModel.from_document(atom_3level_model_document)


@pytest.fixture(name="atom_3level_calibration")
def atom_3level_calibration_fixture(
    atom_3level_calibration_document,
):
    return Atom3LevelCalibration(atom_3level_calibration_document)


@pytest.fixture(name="model", scope="session")
def model_fixture():
    """The shared immutable physics model."""
    return TransmonModel.from_document(_read("sc_transmon_exchange.json"))


@pytest.fixture(name="calibration", scope="session")
def calibration_fixture():
    """The shared immutable portable calibration."""
    return TransmonCalibration(_read("sc_transmon_exchange_calibration.json"))


@pytest.fixture(name="build_model_and_calibration")
def build_model_and_calibration_fixture():
    """Build independent sources for structural-address portability tests."""

    def build():
        built = TransmonModel.from_document(_read("sc_transmon_exchange.json"))
        return built, TransmonCalibration(
            _read("sc_transmon_exchange_calibration.json")
        )

    return build


@pytest.fixture(name="backend")
def backend_fixture(model, calibration) -> TransmonEmulator:
    """A noise-free backend on the shared model/calibration pair.

    Function-scoped: `TransmonEmulator` owns a mutable noise model and private
    copies of the implementation maps, and several tests register noise on it.
    """
    del calibration
    return TransmonEmulator(model, method="density_matrix")
