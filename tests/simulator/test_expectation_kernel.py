"""Expectation kernels against a dense reference operator.

Both kernels avoid building the 2**n x 2**n operator, so every case is checked
against one assembled explicitly with Kronecker products - an independent
computation, not a rearrangement of the same arithmetic.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.observable import Observable
from fatqat.simulator._engine.expectation import (
    expectation_density_matrix,
    expectation_statevector,
)

_N = 5

_LOCAL = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]).astype(complex),
    "ZERO": np.diag([1, 0]).astype(complex),
    "ONE": np.diag([0, 1]).astype(complex),
    "I": np.eye(2, dtype=complex),
}


def _dense(observable: Observable) -> np.ndarray:
    """Assemble the full operator the kernels deliberately never build."""
    total = np.zeros((2**_N, 2**_N), dtype=complex)
    for coefficient, factors in observable.terms:
        letters = dict(factors)
        locals_ = [_LOCAL[letters.get(qubit, "I")] for qubit in range(_N)]
        operator = locals_[0]
        for qubit in range(1, _N):
            operator = np.kron(operator, locals_[qubit])
        total += coefficient * operator
    return total


def _program():
    program = fq.Program(_N)
    for qubit in range(_N):
        program.add(ops.RY(0.3 + 0.2 * qubit), qubit)
    for qubit in range(_N - 1):
        program.add(ops.CX, (qubit, qubit + 1))
    return program


@pytest.fixture(scope="module", name="states")
def _states():
    config = {"counts": False, "final_state": True}
    statevector = (
        fq.simulator.Simulator(method="SV")
        .run(_program(), result_config=config)
        .result()
        .get_statevector()
    )
    density_matrix = (
        fq.simulator.Simulator(method="DM")
        .run(_program(), result_config=config)
        .result()
        .get_density_matrix()
    )
    return statevector, density_matrix


_CASES = {
    "diagonal_zz": Observable.from_sparse([("ZZ", (0, 2), 1.0)], num_qubits=_N),
    "off_diagonal_xy": Observable.from_sparse([("XY", (1, 3), 1.0)], num_qubits=_N),
    "single_y": Observable.from_sparse([("Y", (2,), 1.0)], num_qubits=_N),
    "projector_one": Observable.from_sparse([("ONE", (2,), 1.0)], num_qubits=_N),
    "projector_zero": Observable.from_sparse([("ZERO", (0,), 1.0)], num_qubits=_N),
    "projector_with_pauli": Observable.from_sparse(
        [(["ZERO", "Z"], (0, 4), 1.0)], num_qubits=_N
    ),
    "projector_with_xy": Observable.from_sparse(
        [(["ONE", "X", "Y"], (0, 1, 3), 1.0)], num_qubits=_N
    ),
    "multi_term": Observable.from_sparse(
        [("ZZ", (0, 1), 1.5), ("XX", (2, 3), -0.5), ("ONE", (4,), 0.25)],
        num_qubits=_N,
    ),
    "identity_term": Observable([("I" * _N, 2.5)]),
    "zero_coefficient": Observable([("Z" + "I" * (_N - 1), 0.0)]),
    "repeated_term": Observable([("ZZ" + "I" * (_N - 2), 1.0)] * 2),
}


@pytest.mark.parametrize("observable", _CASES.values(), ids=_CASES.keys())
def test_statevector_kernel_matches_dense_operator(observable, states):
    statevector, _ = states
    reference = complex(np.vdot(statevector, _dense(observable) @ statevector)).real

    assert expectation_statevector(statevector, observable.terms) == pytest.approx(
        reference, abs=1e-12
    )


@pytest.mark.parametrize("observable", _CASES.values(), ids=_CASES.keys())
def test_density_matrix_kernel_matches_dense_operator(observable, states):
    _, density_matrix = states
    reference = np.trace(density_matrix @ _dense(observable)).real

    assert expectation_density_matrix(
        density_matrix, observable.terms
    ) == pytest.approx(reference, abs=1e-12)


def test_occupation_equals_probability_of_bit_set(states):
    # <ONE_i> is the occupation number of qubit i - the quantity atom-array
    # experiments report - so it must equal P(bit i == 1).
    statevector, _ = states
    probabilities = np.abs(statevector) ** 2
    index = np.arange(statevector.size)

    for qubit in range(_N):
        observable = Observable.from_sparse([("ONE", (qubit,), 1.0)], num_qubits=_N)
        expected = probabilities[(index >> (_N - 1 - qubit)) & 1 == 1].sum()

        assert expectation_statevector(statevector, observable.terms) == pytest.approx(
            expected, abs=1e-12
        )


def test_zero_and_one_projectors_are_complementary(states):
    statevector, _ = states
    zero = Observable.from_sparse([("ZERO", (1,), 1.0)], num_qubits=_N)
    one = Observable.from_sparse([("ONE", (1,), 1.0)], num_qubits=_N)

    total = expectation_statevector(statevector, zero.terms) + expectation_statevector(
        statevector, one.terms
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_asymmetric_public_factor_matches_complex_state_and_density_oracles():
    psi = np.asarray([1 + 2j, -3 + 0.5j, 2 - 1j, 0.7 + 1.2j])
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    operator = np.kron(_LOCAL["X"], _LOCAL["I"])
    observable = Observable([("XI", 1.0)])

    assert expectation_statevector(psi, observable.terms) == pytest.approx(
        np.vdot(psi, operator @ psi).real, abs=1e-12
    )
    assert expectation_density_matrix(rho, observable.terms) == pytest.approx(
        np.trace(rho @ operator).real, abs=1e-12
    )


def test_statevector_kernel_does_not_modify_the_state(states):
    statevector, _ = states
    before = statevector.copy()

    expectation_statevector(statevector, _CASES["projector_with_xy"].terms)

    assert np.array_equal(statevector, before)


def test_sign_uses_selection_not_uint8_arithmetic(states):
    # Parity is folded through unsigned intermediates. A Z-heavy term would be
    # wildly wrong while a pure-X term still passed if the sign wrapped, so pin
    # the Z case.
    statevector, _ = states
    observable = Observable([("ZZZZZ", 1.0)])
    reference = complex(np.vdot(statevector, _dense(observable) @ statevector)).real

    value = expectation_statevector(statevector, observable.terms)
    assert value == pytest.approx(reference, abs=1e-12)
    assert abs(value) <= 1.0  # a Pauli expectation cannot leave [-1, 1]
