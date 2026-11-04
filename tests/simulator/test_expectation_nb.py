"""Compiled expectation kernels: agreement with NumPy, and error growth.

The compiled kernels exist only to be faster, so every test here asks whether
they still answer the same question. Two properties matter and are checked
separately: the value must match the NumPy form, and the *error* must not grow
with the state size, which is what the blocked summation is for.
"""

import math

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.observable import NumberOperator, Observable, Projector
from fatqat.simulator._engine.numerics.numba_kernels import _expectation as nbk
from fatqat.simulator._engine.numerics.numpy_kernels import _expectation as npk


def _msb_strides(dims):
    """Place values with subsystem 0 most significant."""
    return [math.prod(dims[index + 1 :]) for index in range(len(dims))]


_CONFIG = {"counts": False, "final_state": True}


def _program(num_qubits, depth=3):
    program = fq.Program(num_qubits)
    for layer in range(depth):
        for qubit in range(num_qubits):
            program.add(ops.RY(0.3 + 0.11 * qubit + 0.4 * layer), qubit)
        for qubit in range(num_qubits - 1):
            program.add(ops.CX, (qubit, qubit + 1))
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
    observable = Observable.from_sparse([(letters, qubits, 1.0)])
    strides = _msb_strides((2,) * num_qubits)
    return npk._term_masks(observable.terms[0][1], strides)[:4]


@pytest.mark.parametrize("case", _TERMS.values(), ids=_TERMS.keys())
def test_statevector_kernel_matches_numpy(case):
    state = _statevector(6)
    index = np.arange(state.shape[0])
    masks = _masks(*case, 6)

    compiled = nbk.qubit_statevector_term(state, *masks)
    reference = npk.qubit_statevector_term(state, index, *masks)

    assert compiled == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("case", _TERMS.values(), ids=_TERMS.keys())
def test_density_matrix_kernel_matches_numpy(case):
    rho = _density_matrix(5)
    index = np.arange(rho.shape[0])
    masks = _masks(*case, 5)

    compiled = nbk.qubit_density_matrix_term(rho, *masks)
    reference = npk.qubit_density_matrix_term(rho, index, *masks)

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
                nbk.qubit_statevector_term(state, *masks)
                - npk.qubit_statevector_term(state, index, *masks)
            ),
        )

    assert worst < 1e-14


def test_estimator_values_are_unchanged_by_the_compiled_path():
    # The integration check: whichever runtime computes it, the public answer is
    # the same one the dense reference gives.
    program = _program(4)
    observable = Observable.from_sparse(
        [("ZZ", (0, 1), 1.5), ("XY", (1, 3), -0.5), (["ONE", "Z"], (2, 0), 0.25)]
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
        "P1": np.diag([0, 1]).astype(complex),
        "I": np.eye(2, dtype=complex),
    }
    dense = np.zeros((16, 16), dtype=complex)
    for coefficient, factors in observable.terms:
        letters = {qubit: str(factor) for qubit, factor in factors}
        operator = local[letters.get(0, "I")]
        for qubit in (1, 2, 3):
            operator = np.kron(operator, local[letters.get(qubit, "I")])
        dense += coefficient * operator
    reference = complex(np.vdot(state, dense @ state)).real

    assert from_estimator == pytest.approx(reference, abs=1e-12)


# Mixed dimensions: one case per digit-kernel factor kind, alone and combined
# with a flip, on place values in both orders. The other subsystems form
# contiguous runs longer than the summation block.
_DIGIT_TERMS = {
    "number": ([NumberOperator()], (2,)),
    "projector": ([Projector(2)], (0,)),
    "x_with_projector": (["X", Projector(3)], (1, 2)),
    "y_with_number": ([NumberOperator(), "Y"], (0, 1)),
    "z_with_number": (["Z", NumberOperator()], (1, 2)),
}
_LAYOUTS = {
    "long_runs": ((3, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2), False),
    "long_runs_little_endian": ((3, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2), True),
}


@pytest.mark.parametrize("layout", _LAYOUTS.values(), ids=_LAYOUTS.keys())
@pytest.mark.parametrize("case", _DIGIT_TERMS.values(), ids=_DIGIT_TERMS.keys())
def test_digit_statevector_kernel_matches_numpy(case, layout):
    dims, little_endian = layout
    strides = (
        [math.prod(dims[:index]) for index in range(len(dims))]
        if little_endian
        else _msb_strides(dims)
    )
    size = math.prod(dims)
    rng = np.random.default_rng(3)
    state = rng.normal(size=size) + 1j * rng.normal(size=size)
    terms = fq.Observable.from_sparse([(*case, 1.0)]).terms

    assert nbk.expectation_statevector(state, terms, dims, strides) == pytest.approx(
        npk.expectation_statevector(state, terms, dims, strides), rel=1e-12
    )


def test_parity_helper_agrees_with_popcount():
    # The kernel needs only popcount & 1, and folds it with XOR shifts rather
    # than counting. Checked against numpy across bit widths, including the
    # high half of a 64-bit mask, where the first fold step matters.
    from fatqat.simulator._engine.numerics.numba_kernels._expectation import _odd_parity

    values = [0, 1, 3, 255, 1 << 20, (1 << 33) | 5, (1 << 62) | (1 << 3) | 1]
    for value in values:
        expected = bool(value.bit_count() & 1)
        assert _odd_parity(value) is expected
