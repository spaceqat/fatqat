"""Tests statevector backend execution, validation, repeatability, and counts."""

import warnings

import numpy as np
import pytest

from fatqat.simulator import Simulator
from fatqat.errors import (
    BackendValidationError,
    MatrixImplementationError,
    UnsupportedOperationError,
)
from fatqat.implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
import fatqat.operations as ops
from fatqat.program import Program


def _h_cz_program():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.measure(0, 0)
    p.measure(1, 1)
    return p


def test_run_with_seed_is_repeatable_and_reinitializes():
    # `seed` is a run kwarg; two runs with the same seed give identical counts,
    # and each run re-initializes (X|0> = |1>, not continuing from leftover |1>).
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.measure(0, 0)
    backend = Simulator("SV")
    first = (
        backend.run(p, shots=10, simulation_config={"seed": 5}).result().get_counts()
    )
    second = (
        backend.run(p, shots=10, simulation_config={"seed": 5}).result().get_counts()
    )
    assert first == second == {"1": 10}


def test_run_without_seed_uses_random_rng_seed(monkeypatch):
    observed = []

    def fake_default_rng(seed=None):
        observed.append(seed)
        return object()

    monkeypatch.setattr(np.random, "default_rng", fake_default_rng)

    p = Program(1)
    p.add(ops.X, 0)
    Simulator("SV").run(p, result_config={"counts": False}).result().get_statevector()

    assert observed == [None]


def test_unsupported_operation_raises():
    class FooGate(ops.Operation):
        name = "FOO"
        num_subsystems = 1

    p = Program(1, 1)
    p.add(FooGate(), 0)
    p.measure(0, 0)
    with pytest.raises(UnsupportedOperationError):
        Simulator("SV").run(p, shots=10)


def test_condition_now_runs():
    # condition reads unwritten slot (0); X applies -> qubit 1 becomes |1>.
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 0))
    p.measure(1, 1)
    with pytest.warns(UserWarning, match="clbits that were never measured"):
        counts = (
            Simulator("SV")
            .run(p, shots=16, simulation_config={"seed": 0})
            .result()
            .get_counts()
        )
    assert counts == {"10": 16}  # c1=1, c0=0 -> "10"


def test_mid_circuit_measurement_now_runs():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.add(ops.X, 1)
    p.measure(1, 1)
    counts = (
        Simulator("SV")
        .run(p, shots=64, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert set(counts) <= {"10", "11"}  # c1 always 1; c0 either


def test_nonpositive_shots_with_counts_raises():
    with pytest.raises(BackendValidationError):
        Simulator("SV").run(_h_cz_program(), shots=0)


@pytest.mark.parametrize("shots", [1.5, True, "10"])
def test_non_integer_shots_with_counts_raises(shots):
    with pytest.raises(BackendValidationError):
        Simulator("SV").run(_h_cz_program(), shots=shots)


@pytest.mark.parametrize("shots", [0, -1, 2])
def test_measured_statevector_requires_exactly_one_shot(shots):
    with pytest.raises(BackendValidationError):
        Simulator("SV").run(
            _h_cz_program(),
            shots=shots,
            result_config={"counts": False, "final_state": True},
        )


def test_deterministic_with_seed():
    a = (
        Simulator("SV")
        .run(_h_cz_program(), shots=300, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )
    b = (
        Simulator("SV")
        .run(_h_cz_program(), shots=300, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )
    assert a == b


def test_grouped_measurements_do_not_warn_about_unmeasured_clbits():
    p = Program(2, 2)
    p.add(ops.X, 0)
    p.add(ops.X, 1)
    p.measure((0, 1), (0, 1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts = (
            Simulator("SV")
            .run(p, shots=4, simulation_config={"seed": 0})
            .result()
            .get_counts()
        )

    assert counts == {"11": 4}
    assert not caught


def test_rule_failure_is_wrapped_with_operation_context():
    def broken_rule(op):
        raise RuntimeError("boom")

    m = MatrixImplementationMap()
    m.add(ops.X, broken_rule)
    backend = Simulator("SV", implementation_map=m)

    p = Program(1, 1)
    p.add(ops.X, 0)
    p.measure(0, 0)

    with pytest.raises(MatrixImplementationError, match="XGate") as excinfo:
        backend.run(p, shots=10)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "boom"


def test_custom_operation_runs_end_to_end_via_bare_callable():
    class MyX(ops.Operation):
        name = "MyX"
        num_subsystems = 1

    def my_x_rule(op):
        return np.array([[0, 1], [1, 0]], dtype=complex)

    m = default_matrix_implementation_map()
    m.add(MyX, my_x_rule)
    backend = Simulator("SV", implementation_map=m)

    p = Program(1)
    p.add(MyX(), 0)

    statevector = (
        backend.run(p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(statevector, [0, 1])


def test_unregistered_gate_raises_after_remove():
    m = default_matrix_implementation_map()
    m.remove(ops.T)
    backend = Simulator("SV", implementation_map=m)

    p = Program(1, 1)
    p.add(ops.T, 0)
    p.measure(0, 0)

    with pytest.raises(UnsupportedOperationError):
        backend.run(p, shots=10)
