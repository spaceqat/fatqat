"""Tests statevector backend execution, validation, repeatability, and counts."""

import pytest

import qnsim as qs
from qnsim.backends import StateVectorBackend
from qnsim.errors import BackendValidationError, UnsupportedOperationError
from qnsim import operations as ops
from qnsim.program import Program


def _h_cz_program():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    return p


def test_backend_run_is_repeatable():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    backend = StateVectorBackend()
    first = backend.run(p, shots=10, seed=0).result().get_counts()
    second = backend.run(p, shots=10, seed=0).result().get_counts()
    # X|0> = |1>; second run must re-initialize, not continue from leftover |1>
    assert first == {"1": 10}
    assert second == {"1": 10}


def test_seed_is_a_run_kwarg_and_repeatable():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    backend = StateVectorBackend()
    a = backend.run(p, shots=10, seed=5).result().get_counts()
    b = backend.run(p, shots=10, seed=5).result().get_counts()
    assert a == b == {"1": 10}


def test_unsupported_operation_raises():
    class FooGate(ops.Operation):
        name = "FOO"
        _num_qubits = 1

    p = Program(1, 1)
    p.add(FooGate(), 0)
    p.add_measurement(0, 0)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_condition_rejected():
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 1))
    p.add_measurement(0, 0)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_mid_circuit_measurement_rejected():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1)  # gate after a measurement
    p.add_measurement(1, 1)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_nonpositive_shots_with_counts_raises():
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(_h_cz_program(), shots=0)


@pytest.mark.parametrize("shots", [1.5, True, "10"])
def test_non_integer_shots_with_counts_raises(shots):
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(_h_cz_program(), shots=shots)


@pytest.mark.parametrize("shots", [0, -1, 2])
def test_measured_statevector_requires_exactly_one_shot(shots):
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(
            _h_cz_program(),
            shots=shots,
            result_config=qs.ResultConfig(counts=False, statevector=True),
        )


def test_deterministic_with_seed():
    a = StateVectorBackend().run(_h_cz_program(), shots=300, seed=7).result().get_counts()
    b = StateVectorBackend().run(_h_cz_program(), shots=300, seed=7).result().get_counts()
    assert a == b
