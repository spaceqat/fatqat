"""Compiled expectation kernels: agreement with NumPy, and error growth.

The compiled kernels exist only to be faster, so every test here asks whether
they still answer the same question. Two properties matter and are checked
separately: the value must match the NumPy form, and the *error* must not grow
with the state size, which is what the blocked summation is for.
"""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as op
from fatqat.observable import Observable
from fatqat.simulator._engine import expectation as ex

pytest.importorskip("numba")


def test_compiled_kernel_is_loaded_when_numba_is_installed():
    """Guard the rest of this file: numba is here, so the kernel must be too.

    Deliberately an assertion rather than a skip condition. If the compiled
    kernel failed to load, every other test below would be comparing NumPy
    against NumPy - passing while testing nothing. That is the one failure mode
    this file exists to catch, so it has to be loud.
    """
    assert ex.USING_COMPILED_KERNEL, (
        "numba is installed but the compiled expectation kernel did not load, "
        "so the tests below would silently compare NumPy against itself"
    )


_CONFIG = {"counts": False, "final_state": True}


def _program(num_qubits, depth=3):
    program = fq.Program(num_qubits)
    for layer in range(depth):
        for qubit in range(num_qubits):
            program.add(op.RY(0.3 + 0.11 * qubit + 0.4 * layer), qubit)
        for qubit in range(num_qubits - 1):
            program.add(op.CX, (qubit, qubit + 1))
    return program


def _statevector(num_qubits):
    return (
        fq.simulator.Simulator(method="SV")
        .run(_program(num_qubits), shots=0, result_config=_CONFIG)
        .result()
        .get_statevector()
    )


def _density_matrix(num_qubits):
    return (
        fq.simulator.Simulator(method="DM")
        .run(_program(num_qubits), shots=0, result_config=_CONFIG)
        .result()
        .get_density_matrix()
    )


# One case per structural shape the kernel branches on, not per letter: a
# diagonal term (x_mask empty), an off-diagonal one, a Y phase, projectors
# alone, and projectors mixed with Paulis.
_TERMS = {
    "diagonal_zz": ("ZZ", (0, 2)),
    "off_diagonal_xx": ("XX", (1, 3)),
    "with_y_phase": ("XY", (0, 3)),
    "y_only": ("Y", (2,)),
    "projector_one": ("ONE", (1,)),
    "projector_zero": ("ZERO", (3,)),
    "projector_with_pauli": (["ONE", "Z", "X"], (0, 2, 3)),
    "identity": ((), ()),
}


def _masks(letters, qubits, num_qubits):
    if not qubits:
        return (0, 0, 0, 0)
    observable = Observable.from_sparse([(letters, qubits, 1.0)], num_qubits=num_qubits)
    return ex._term_masks(observable.terms[0][1])[:4]


@pytest.mark.parametrize("case", _TERMS.values(), ids=_TERMS.keys())
def test_statevector_kernel_matches_numpy(case):
    state = _statevector(6)
    index = np.arange(state.shape[0])
    masks = _masks(*case, 6)

    compiled = ex._COMPILED[0](state, *masks)
    reference = ex._statevector_term_numpy(state, index, *masks)

    assert compiled == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("case", _TERMS.values(), ids=_TERMS.keys())
def test_density_matrix_kernel_matches_numpy(case):
    rho = _density_matrix(5)
    index = np.arange(rho.shape[0])
    masks = _masks(*case, 5)

    compiled = ex._COMPILED[1](rho, *masks)
    reference = ex._density_matrix_term_numpy(rho, index, *masks)

    assert compiled == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("num_qubits", [4, 8, 12, 16])
def test_agreement_does_not_degrade_with_state_size(num_qubits):
    # The point of summing per block. A flat running sum drifts as O(N * eps),
    # so this margin would be crossed somewhere past a million amplitudes while
    # every small case still passed; blocked summation keeps it flat in N.
    state = _statevector(num_qubits)
    index = np.arange(state.shape[0])
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(12):
        qubits = tuple(int(q) for q in rng.choice(num_qubits, 2, replace=False))
        masks = _masks("XZ", qubits, num_qubits)
        worst = max(
            worst,
            abs(
                ex._COMPILED[0](state, *masks)
                - ex._statevector_term_numpy(state, index, *masks)
            ),
        )

    assert worst < 1e-14


def test_estimator_values_are_unchanged_by_the_compiled_path():
    # The integration check: whichever kernel is loaded, the public answer is
    # the same one the dense reference gives.
    program = _program(4)
    observable = Observable.from_sparse(
        [("ZZ", (0, 1), 1.5), ("XY", (1, 3), -0.5), (["ONE", "Z"], (2, 0), 0.25)],
        num_qubits=4,
    )
    state = _statevector(4)

    from_estimator = (
        fq.Estimator(fq.simulator.Simulator(method="SV"))
        .run(program, observable)
        .result()
        .get_expectation()
    )

    local = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]).astype(complex),
        "ONE": np.diag([0, 1]).astype(complex),
        "I": np.eye(2, dtype=complex),
    }
    dense = np.zeros((16, 16), dtype=complex)
    for coefficient, factors in observable.terms:
        letters = dict(factors)
        operator = local[letters.get(3, "I")]
        for qubit in (2, 1, 0):
            operator = np.kron(operator, local[letters.get(qubit, "I")])
        dense += coefficient * operator
    reference = complex(np.vdot(state, dense @ state)).real

    assert from_estimator == pytest.approx(reference, abs=1e-12)


def test_parity_helper_agrees_with_popcount():
    # The kernel needs only popcount & 1, and folds it with XOR shifts rather
    # than counting. Checked against numpy across bit widths, including the
    # high half of a 64-bit mask, where the first fold step matters.
    from fatqat.simulator._engine.expectation_nb import _odd_parity

    values = [0, 1, 3, 255, 1 << 20, (1 << 33) | 5, (1 << 62) | (1 << 3) | 1]
    for value in values:
        expected = bool(value.bit_count() & 1)
        assert _odd_parity(value) is expected
