"""The pulse emulator's own exact-expectation evaluator.

The evaluator numbers subsystems the way QuTiP does, with subsystem 0 the most
significant tensor factor. Every case is checked against an operator assembled
with Kronecker products in that order.
"""

import numpy as np
import pytest
import qutip

from fatqat.emulator._expectation import expectation, observable_operator
from fatqat.observable import NumberOperator, Observable, Projector

_DIMS = (3, 2, 4)
_PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]).astype(complex),
}

_CASES = {
    "number": Observable.from_sparse([(NumberOperator(), (2,), 1.0)]),
    "level_projector": Observable.from_sparse([(Projector(2), (0,), 1.0)]),
    "number_product": Observable.from_sparse(
        [([NumberOperator(), NumberOperator()], (0, 2), 1.0)]
    ),
    "projector_with_x": Observable.from_sparse([([Projector(1), "X"], (0, 1), 1.0)]),
    "y_with_number": Observable.from_sparse([(["Y", NumberOperator()], (1, 2), 1.0)]),
    "z_with_projector": Observable.from_sparse([(["Z", Projector(3)], (1, 2), 1.0)]),
    "multi_term": Observable.from_sparse(
        [
            (NumberOperator(), (0,), 0.5),
            ("X", (1,), -1.5),
            ([Projector(0), "Y", Projector(3)], (0, 1, 2), 2.0),
            ([], (), 0.25),
        ]
    ),
}


def _local(factor, dim):
    if isinstance(factor, NumberOperator):
        return np.diag(np.arange(dim)).astype(complex)
    if isinstance(factor, Projector):
        local = np.zeros((dim, dim), dtype=complex)
        local[factor.level, factor.level] = 1.0
        return local
    return _PAULIS[str(factor)]


def _dense(observable):
    total = np.zeros((24, 24), dtype=complex)
    for coefficient, factors in observable.terms:
        locals_ = [np.eye(dim, dtype=complex) for dim in _DIMS]
        for index, factor in factors:
            locals_[index] = _local(factor, _DIMS[index])
        operator = locals_[0]
        for local in locals_[1:]:
            operator = np.kron(operator, local)
        total += coefficient * operator
    return total


def _value(state, observable):
    return expectation(state, observable_operator(observable.terms, _DIMS))


@pytest.fixture(scope="module", name="states")
def _states():
    rng = np.random.default_rng(7)
    psi = rng.normal(size=24) + 1j * rng.normal(size=24)
    psi /= np.linalg.norm(psi)
    vectors = rng.normal(size=(24, 3)) + 1j * rng.normal(size=(24, 3))
    rho = vectors @ vectors.conj().T
    dims = list(_DIMS)
    return (
        qutip.Qobj(psi.reshape(-1, 1), dims=[dims, [1] * len(dims)]),
        qutip.Qobj(rho / np.trace(rho), dims=[dims, dims]),
    )


@pytest.mark.parametrize("observable", _CASES.values(), ids=_CASES.keys())
def test_statevector_matches_dense_operator(observable, states):
    psi, _ = states

    vector = psi.full().reshape(-1)

    assert _value(psi, observable) == pytest.approx(
        np.vdot(vector, _dense(observable) @ vector).real, abs=1e-12
    )


@pytest.mark.parametrize("observable", _CASES.values(), ids=_CASES.keys())
def test_density_matrix_matches_dense_operator(observable, states):
    _, rho = states

    assert _value(rho, observable) == pytest.approx(
        np.trace(rho.full() @ _dense(observable)).real, abs=1e-12
    )
