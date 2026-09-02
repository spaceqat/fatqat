import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import (
    SCQubitGoogleSimulator,
    SCQubitIBMSimulator,
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
from fatqat.program import _AppliedOperation
from fatqat.resource_layout import ResourceLayout
from fatqat.result import _ResultConfig

# Both concrete SC simulators expose the same small constructor contract,
# while implementing their own target-facing site property.
_SC_BACKENDS = (SCQubitIBMSimulator, SCQubitGoogleSimulator)


@pytest.mark.parametrize("backend_cls", _SC_BACKENDS)
def test_sc_backend_rejects_non_positive_num_qubits(backend_cls):
    with pytest.raises(ValueError, match="positive"):
        backend_cls(num_qubits=0)


@pytest.mark.parametrize("backend_cls", _SC_BACKENDS)
def test_sc_backend_rejects_coupling_outside_device_sites(backend_cls):
    with pytest.raises(ValueError, match="outside"):
        backend_cls(num_qubits=3, couplings=((0, 3),))


@pytest.mark.parametrize("backend_cls", _SC_BACKENDS)
def test_sc_backend_rejects_self_coupling(backend_cls):
    with pytest.raises(ValueError, match="distinct"):
        backend_cls(num_qubits=3, couplings=((1, 1),))


@pytest.mark.parametrize("backend_cls", _SC_BACKENDS)
def test_sc_backend_rejects_non_integer_coupling_endpoint(backend_cls):
    with pytest.raises(TypeError, match="integers"):
        backend_cls(num_qubits=3, couplings=((0, "1"),))


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
    program.add(ops.Reset, (0, 2), condition=(0, 1))
    step = next(
        instruction
        for instruction in program._instructions
        if isinstance(instruction, _AppliedOperation)
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
        instr for instr in program._instructions if isinstance(instr, ops.Measurement)
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
