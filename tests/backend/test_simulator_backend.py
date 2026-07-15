"""Tests SimulatorBackend method selection."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.errors import BackendValidationError
from fatqat import operations as ops
from fatqat.program import Program


def _bell() -> Program:
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.add_measurement((0, 1), (0, 1))
    return p


# --- method selection --------------------------------------------------------


def test_default_method_is_statevector():
    backend = SimulatorBackend()
    assert backend._state_field == "statevector"


@pytest.mark.parametrize("alias", ["density_matrix", "DM", "dm", "Dm"])
def test_density_matrix_aliases(alias):
    assert SimulatorBackend(method=alias)._state_field == "density_matrix"


@pytest.mark.parametrize("alias", ["statevector", "SV", "sv"])
def test_statevector_aliases(alias):
    assert SimulatorBackend(method=alias)._state_field == "statevector"


def test_unknown_method_rejected():
    with pytest.raises(BackendValidationError, match="unsupported method"):
        SimulatorBackend(method="mps")


def test_alias_selects_identical_behavior():
    p = _bell()
    a = SimulatorBackend(method="SV").run(p, shots=64, seed=7).result()
    b = SimulatorBackend(method="statevector").run(p, shots=64, seed=7).result()
    assert a.get_counts() == b.get_counts()
    c = SimulatorBackend(method="DM").run(p, shots=64, seed=7).result()
    d = SimulatorBackend(method="density_matrix").run(p, shots=64, seed=7).result()
    assert c.get_counts() == d.get_counts()


def test_method_selects_native_state_field():
    p = Program(1)
    p.add(ops.H, 0)
    sv = (
        SimulatorBackend(method="SV")
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    rho = (
        SimulatorBackend(method="DM")
        .run(p, result_config={"counts": False, "density_matrix": True})
        .result()
        .get_density_matrix()
    )
    assert np.allclose(rho, np.outer(sv, sv.conj()))


def test_metadata_records_method_and_backend_name():
    p = _bell()
    result = SimulatorBackend(method="DM").run(p, shots=5, seed=0).result()
    assert result.metadata["backend_name"] == "SimulatorBackend"
    assert result.metadata["method"] == "density_matrix"
