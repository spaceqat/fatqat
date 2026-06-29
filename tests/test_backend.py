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


def test_counts_happy_path_keys():
    backend = StateVectorBackend(seed=123)
    job = backend.run(_h_cz_program(), shots=500, result_config=qs.ResultConfig(counts=True))
    counts = job.result().get_counts()
    assert sum(counts.values()) == 500
    # H on q0 then CZ (no effect here): q1 always 0, q0 ~ 50/50 -> keys "00"/"01"
    assert set(counts) <= {"00", "01"}


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


def test_deterministic_with_seed():
    a = StateVectorBackend(seed=7).run(_h_cz_program(), shots=300).result().get_counts()
    b = StateVectorBackend(seed=7).run(_h_cz_program(), shots=300).result().get_counts()
    assert a == b
