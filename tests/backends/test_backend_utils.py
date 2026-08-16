import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import (
    AtomGridSimulator,
    SCQubitGoogleSimulator,
    SCQubitIBMSimulator,
)
from fatqat.simulator.fake_atom_grid import fake_atom_grid_implementation_map
from fatqat.simulator.fake_superconducting import (
    fake_superconducting_google_implementation_map,
    fake_superconducting_ibm_implementation_map,
)
from fatqat._index_allocation import _ClassicalAllocation, _EngineAllocation
from fatqat._backends.backend_utils import (
    _lower_measurement_boundary,
    _lower_reset_boundary,
    _resolve_result_flags,
    _validate_result_shots,
)
from fatqat._backends.steps import ResetStep
from fatqat.errors import BackendValidationError
from fatqat.noise import NoiseModel, ReadoutConfusion
from fatqat.program import AppliedOperation
from fatqat.resource_layout import ResourceLayout
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


def test_resolve_result_flags_defaults_state_for_nonstochastic_program():
    flags = _resolve_result_flags(
        _ResultConfig(counts=None, final_state=None),
        has_measurement=False,
        stochastic_final_state=False,
    )

    assert flags == (False, True)


def test_resolve_result_flags_stochastic_state_defaults_off():
    flags = _resolve_result_flags(
        _ResultConfig(counts=None, final_state=None),
        has_measurement=False,
        stochastic_final_state=True,
    )

    assert flags == (False, False)


def test_resolve_result_flags_preserves_explicit_overrides():
    flags = _resolve_result_flags(
        _ResultConfig(counts=False, final_state=True),
        has_measurement=True,
        stochastic_final_state=True,
    )

    assert flags == (False, True)


def test_validate_result_shots_uses_complete_caller_owned_messages():
    with pytest.raises(BackendValidationError) as exc:
        _validate_result_shots(
            counts=True,
            explicit_final_state=False,
            stochastic_final_state=False,
            shots=1.5,
            shots_type_error="family-specific integer message",
            state_label="statevector",
            stochastic_sources="measurement",
        )
    assert str(exc.value) == "family-specific integer message"

    with pytest.raises(BackendValidationError) as exc:
        _validate_result_shots(
            counts=True,
            explicit_final_state=False,
            stochastic_final_state=False,
            shots=0,
            shots_type_error="unused",
            state_label="statevector",
            stochastic_sources="measurement",
        )
    assert str(exc.value) == "counts require shots > 0, got shots=0"

    with pytest.raises(BackendValidationError) as exc:
        _validate_result_shots(
            counts=False,
            explicit_final_state=True,
            stochastic_final_state=True,
            shots=2,
            shots_type_error="unused",
            state_label="statevector",
            stochastic_sources="measurement",
        )
    assert str(exc.value) == (
        "statevector with measurement is only supported for shots == 1"
    )


# --- _lower_reset_boundary: shared by the matrix and pulse families ---


def test_reset_boundary_resolves_all_targets_and_condition():
    program = fq.Program(3, 1)
    program.add(fq.ops.Reset, (0, 2), condition=(0, 1))
    step = next(
        instruction
        for instruction in program.operations
        if isinstance(instruction, AppliedOperation)
    )
    layout = ResourceLayout({program.quantum_registers[0][i]: i for i in range(3)})
    engine_allocation = _EngineAllocation((0, 1, 2), (2, 2, 2))
    classical_allocation = _ClassicalAllocation.from_program(program)

    assert _lower_reset_boundary(
        step, layout, engine_allocation, classical_allocation
    ) == ResetStep(
        reset_indices=(0, 2),
        condition=((0, 1),),
    )


# --- _lower_measurement_boundary: shared by the matrix and pulse families --


def _one_qubit_measurement_setup(confusion, *, reported_digit_map):
    program = fq.Program(1, 1)
    program.measure(0, 0)
    (step,) = [
        instr
        for instr in program.operations
        if isinstance(instr, fq.operations.Measurement)
    ]
    q0 = program.quantum_registers[0][0]
    resource_layout = ResourceLayout({q0: 0})
    engine_allocation = _EngineAllocation((0,), (2,))
    classical_allocation = _ClassicalAllocation.from_program(program)
    noise = NoiseModel()
    if confusion is not None:
        noise.add(ReadoutConfusion(confusion), targets=q0)
    return (
        step,
        (reported_digit_map,),
        resource_layout,
        engine_allocation,
        classical_allocation,
        noise,
    )


def test_boundary_resolves_indices_and_collapses_no_confusion_to_none():
    step, maps, layout, engine, classical, noise = _one_qubit_measurement_setup(
        None, reported_digit_map=(0, 1)
    )
    measured, classical, confusions = _lower_measurement_boundary(
        step, maps, layout, engine, classical, noise
    )
    assert measured == (0,)
    assert classical == (0,)
    assert confusions is None


def test_boundary_validates_confusion_shape_against_the_callers_reported_map():
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    step, maps, layout, engine, classical, noise = _one_qubit_measurement_setup(
        always_flip, reported_digit_map=(0, 1)
    )
    _measured, _classical, confusions = _lower_measurement_boundary(
        step, maps, layout, engine, classical, noise
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
        step, maps, layout, engine, classical, noise = _one_qubit_measurement_setup(
            mismatched, reported_digit_map=reported_digit_map
        )
        with pytest.raises(
            BackendValidationError, match="reported classical dimension"
        ):
            _lower_measurement_boundary(step, maps, layout, engine, classical, noise)
