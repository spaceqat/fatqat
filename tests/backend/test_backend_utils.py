import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import AtomGridBackend, SCQubitGoogleSimulator, SCQubitIBMSimulator
from fatqat.backends.fake_atom_grid import fake_atom_grid_implementation_map
from fatqat.backends.fake_superconducting import (
    fake_superconducting_google_implementation_map,
    fake_superconducting_ibm_implementation_map,
)
from fatqat._engine_index_allocation import _EngineIndexAllocation
from fatqat.backends.backend_utils import (
    _PlanFacts,
    _lower_measurement_boundary,
)
from fatqat.backends.engine_contract import (
    _DensityMatrixResultRequest,
    _StateVectorResultRequest,
)
from fatqat.backends.simulator_backend import _resolve_result_request
from fatqat.errors import BackendValidationError
from fatqat.noise import NoiseModel
from fatqat.resource_layout import ResourceLayout
from fatqat.result import _ResultConfig

# Every fake-target backend constructor routes grid_size through the shared
# `_validate_grid_size` (see fake_atom_grid.py / fake_superconducting.py), so
# this pins that routing for all three rather than just the atom-grid backend.
_GRID_BACKENDS = (AtomGridBackend, SCQubitIBMSimulator, SCQubitGoogleSimulator)


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


# --- _lower_measurement_boundary: shared by the matrix and pulse families --


def _one_qubit_measurement_setup(confusion, *, reported_digit_map):
    program = fq.Program(1, 1)
    program.add_measurement(0, 0)
    (step,) = [
        instr for instr in program.operations if isinstance(instr, fq.operations.Measurement)
    ]
    q0 = program.qreg[0][0]
    resource_layout = ResourceLayout({q0: 0})
    allocation = _EngineIndexAllocation.from_program(program)
    noise = NoiseModel()
    if confusion is not None:
        noise.add_readout_error(confusion, target=q0)
    return step, (reported_digit_map,), resource_layout, allocation, noise


def test_boundary_resolves_indices_and_collapses_no_confusion_to_none():
    step, maps, layout, allocation, noise = _one_qubit_measurement_setup(
        None, reported_digit_map=(0, 1)
    )
    measured, classical, confusions = _lower_measurement_boundary(
        step, maps, layout, allocation, noise
    )
    assert measured == (0,)
    assert classical == (0,)
    assert confusions is None


def test_boundary_validates_confusion_shape_against_the_callers_reported_map():
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    step, maps, layout, allocation, noise = _one_qubit_measurement_setup(
        always_flip, reported_digit_map=(0, 1)
    )
    _measured, _classical, confusions = _lower_measurement_boundary(
        step, maps, layout, allocation, noise
    )
    assert np.array_equal(confusions[0], always_flip)


def test_boundary_rejects_mismatched_confusion_shape_for_any_callers_map():
    # A matrix-style identity map (0, 1) and a pulse-style literal map
    # (0, 1, 1) both imply reported dimension 2 (max(map) + 1), so the same
    # shared check rejects a 3x3 confusion matrix for either caller with
    # the same message - this is the equivalence the task brief calls out:
    # pulse's old hand-rolled "!= (2, 2)" check is exactly this general
    # check's result for its (0, 1, 1) map.
    mismatched = np.eye(3)
    for reported_digit_map in ((0, 1), (0, 1, 1)):
        step, maps, layout, allocation, noise = _one_qubit_measurement_setup(
            mismatched, reported_digit_map=reported_digit_map
        )
        with pytest.raises(BackendValidationError, match="reported classical dimension"):
            _lower_measurement_boundary(step, maps, layout, allocation, noise)
