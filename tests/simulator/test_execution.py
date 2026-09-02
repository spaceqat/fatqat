"""Tests statevector backend execution, validation, repeatability, and counts."""

import warnings
from itertools import product

import numpy as np
import pytest

import fatqat as fq
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


def _monomial_unitary(permutation, angles):
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    matrix[np.asarray(permutation), np.arange(len(permutation))] = np.exp(
        1j * np.asarray(angles)
    )
    return matrix


class FullRegisterGate(ops.Operation):
    name = "FullRegisterGate"
    num_subsystems = 2


FULL_REGISTER_MATRIX = _monomial_unitary([2, 0, 3, 1], [0.17, -0.31, 0.53, 0.89])
PUBLIC_PSI = np.asarray([1 + 2j, -3 + 0.5j, 2 - 1j, 0.7 + 1.2j])
PUBLIC_PSI = PUBLIC_PSI / np.linalg.norm(PUBLIC_PSI)


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
    assert counts == {"01": 16}  # c0=0, c1=1


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
    assert set(counts) <= {"01", "11"}  # c0 varies; c1 is always 1


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


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("target, expected_index", [(0, 2), (1, 1)])
def test_public_qubit_zero_is_the_most_significant_state_digit(
    runtime, target, expected_index
):
    if runtime == "numba":
        pytest.importorskip("numba")
    program = Program(2)
    program.add(ops.X, target)
    state = (
        Simulator("SV", runtime=runtime)
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    expected = np.zeros(4, dtype=complex)
    expected[expected_index] = 1.0
    assert np.array_equal(state, expected)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_full_register_matrix_uses_public_basis_without_conversion(runtime):
    if runtime == "numba":
        pytest.importorskip("numba")
    implementation_map = MatrixImplementationMap()
    implementation_map.add(FullRegisterGate, FULL_REGISTER_MATRIX)
    program = Program(2)
    program.add(FullRegisterGate(), (0, 1))

    state = (
        Simulator("SV", runtime=runtime, implementation_map=implementation_map)
        .run(
            program,
            initial_state=PUBLIC_PSI,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_statevector()
    )
    unitary = (
        Simulator("unitary", runtime=runtime, implementation_map=implementation_map)
        .run(program)
        .result()
        .get_unitary()
    )
    assert np.allclose(state, FULL_REGISTER_MATRIX @ PUBLIC_PSI, atol=1e-12)
    assert np.allclose(unitary, FULL_REGISTER_MATRIX, atol=1e-12)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_mixed_radix_public_state_and_measurement_round_trip(runtime):
    if runtime == "numba":
        pytest.importorskip("numba")
    q0 = fq.QuantumRegister(1, dim=2)
    q1 = fq.QuantumRegister(1, dim=3)
    c0 = fq.ClassicalRegister(1, dim=2)
    c1 = fq.ClassicalRegister(1, dim=3)

    for a0, a1 in product(range(2), range(3)):
        state_program = Program([q0, q1])
        state_program.add(ops.Shift(a0), q0[0])
        state_program.add(ops.Shift(a1), q1[0])
        state = (
            Simulator("SV", runtime=runtime)
            .run(
                state_program,
                result_config={"counts": False, "final_state": True},
            )
            .result()
            .get_statevector()
        )
        expected = np.zeros(6, dtype=complex)
        expected[3 * a0 + a1] = 1.0
        assert np.array_equal(state, expected)

        measured = Program([q0, q1], [c0, c1])
        measured.add(ops.Shift(a0), q0[0])
        measured.add(ops.Shift(a1), q1[0])
        measured.measure((q0[0], q1[0]), (c0[0], c1[0]))
        counts = (
            Simulator("SV", runtime=runtime)
            .run(measured, shots=3)
            .result()
            .get_counts_as_tuples()
        )
        assert counts == {(a0, a1): 3}


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_mixed_radix_full_register_matrix_keeps_operand_dimension_pairs(runtime):
    if runtime == "numba":
        pytest.importorskip("numba")

    class MixedRegisterGate(ops.Operation):
        name = "MixedRegisterGate"
        num_subsystems = 2

    matrix = _monomial_unitary([2, 5, 1, 4, 0, 3], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    psi = np.asarray([1 + 1j, -2 + 0.5j, 0.3 - 1j, 2.1j, -0.7, 1.2 + 0.4j])
    psi = psi / np.linalg.norm(psi)
    q0 = fq.QuantumRegister(1, dim=2)
    q1 = fq.QuantumRegister(1, dim=3)
    program = Program([q0, q1])
    program.add(MixedRegisterGate(), (q0[0], q1[0]))
    implementation_map = MatrixImplementationMap()
    implementation_map.add(MixedRegisterGate, matrix)

    state = (
        Simulator("SV", runtime=runtime, implementation_map=implementation_map)
        .run(
            program,
            initial_state=psi,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_statevector()
    )
    unitary = (
        Simulator("unitary", runtime=runtime, implementation_map=implementation_map)
        .run(program)
        .result()
        .get_unitary()
    )
    assert np.allclose(state, matrix @ psi, atol=1e-12)
    assert np.allclose(unitary, matrix, atol=1e-12)


def test_custom_matrix_uses_first_target_as_local_most_significant_subsystem():
    class RotateLastTarget(ops.Operation):
        name = "RotateLastTarget"
        num_subsystems = 2

    matrix = np.eye(4, dtype=complex)
    matrix[0, 0] = matrix[1, 1] = 0.0
    matrix[0, 1] = matrix[1, 0] = -1j
    implementation_map = MatrixImplementationMap()
    implementation_map.add(RotateLastTarget, matrix)

    for targets, expected_index in [((0, 1), 1), ((1, 0), 2)]:
        program = Program(2)
        program.add(RotateLastTarget(), targets)
        statevector = (
            Simulator("SV", runtime="numpy", implementation_map=implementation_map)
            .run(program, result_config={"counts": False, "final_state": True})
            .result()
            .get_statevector()
        )

        # Local index |01> changes the last member of the ordered target tuple.
        expected = np.zeros(4, dtype=complex)
        expected[expected_index] = -1j
        assert np.allclose(statevector, expected)


def test_unregistered_gate_raises_after_remove():
    m = default_matrix_implementation_map()
    m.remove(ops.T)
    backend = Simulator("SV", implementation_map=m)

    p = Program(1, 1)
    p.add(ops.T, 0)
    p.measure(0, 0)

    with pytest.raises(UnsupportedOperationError):
        backend.run(p, shots=10)
