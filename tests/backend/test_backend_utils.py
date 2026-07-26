import pytest

from fatqat.backends import (
    AtomGridSimulator,
    SCQubitGoogleSimulator,
    SCQubitIBMSimulator,
)
from fatqat.backends.fake_atom_grid import fake_atom_grid_implementation_map
from fatqat.backends.fake_superconducting import (
    fake_superconducting_google_implementation_map,
    fake_superconducting_ibm_implementation_map,
)
from fatqat.backends.backend_utils import _PlanFacts
from fatqat.backends.engine_contract import (
    _DensityMatrixResultRequest,
    _StateVectorResultRequest,
)
from fatqat.backends.simulator_backend import _resolve_result_request
from fatqat.result import _ResultConfig

# Every fake-target backend constructor routes grid_size through the shared
# `_validate_grid_size` (see fake_atom_grid.py / fake_superconducting.py), so
# this pins that routing for all three rather than just the atom-grid backend.
_GRID_BACKENDS = (AtomGridSimulator, SCQubitIBMSimulator, SCQubitGoogleSimulator)


@pytest.mark.parametrize("backend_cls", _GRID_BACKENDS)
def test_backend_rejects_non_tuple_grid_size(backend_cls):
    with pytest.raises(TypeError):
        backend_cls(grid_size=[2, 3])


@pytest.mark.parametrize("backend_cls", _GRID_BACKENDS)
def test_backend_rejects_grid_size_with_wrong_length(backend_cls):
    with pytest.raises(ValueError):
        backend_cls(grid_size=(2,))


@pytest.mark.parametrize("backend_cls", _GRID_BACKENDS)
def test_backend_rejects_non_int_grid_entry(backend_cls):
    with pytest.raises(TypeError):
        backend_cls(grid_size=(2, "3"))


@pytest.mark.parametrize("backend_cls", _GRID_BACKENDS)
def test_backend_rejects_bool_grid_entry(backend_cls):
    with pytest.raises(TypeError):
        backend_cls(grid_size=(True, 3))


@pytest.mark.parametrize("backend_cls", _GRID_BACKENDS)
def test_backend_rejects_non_positive_grid_entry(backend_cls):
    with pytest.raises(ValueError):
        backend_cls(grid_size=(0, 3))


@pytest.mark.parametrize(
    "implementation_map",
    (
        fake_atom_grid_implementation_map,
        fake_superconducting_ibm_implementation_map,
        fake_superconducting_google_implementation_map,
    ),
)
def test_grid_implementation_map_rejects_invalid_shape(implementation_map):
    with pytest.raises(ValueError):
        implementation_map(0, 3)


def test_resolve_result_request_defaults_statevector_for_nonstochastic_program():
    request = _resolve_result_request(
        _ResultConfig(counts=None, final_state=None),
        _PlanFacts(has_measurement=False, has_reset=False),
        _StateVectorResultRequest,
        "statevector",
        nonunitary_is_stochastic=True,
    )

    assert request.counts is False
    assert request.statevector is True


def test_resolve_result_request_reset_suppresses_statevector_default():
    request = _resolve_result_request(
        _ResultConfig(counts=None, final_state=None),
        _PlanFacts(has_measurement=False, has_reset=True),
        _StateVectorResultRequest,
        "statevector",
        nonunitary_is_stochastic=True,
    )

    assert request.statevector is False


def test_resolve_result_request_reset_keeps_density_matrix_default():
    request = _resolve_result_request(
        _ResultConfig(counts=None, final_state=None),
        _PlanFacts(has_measurement=False, has_reset=True),
        _DensityMatrixResultRequest,
        "density_matrix",
        nonunitary_is_stochastic=False,
    )

    assert request.counts is False
    assert request.density_matrix is True
