"""Behavioral equivalence of `NumbaEngine` with the base `NumpyEngine`.

The Numba engine swaps only the statevector matrix-application kernel via a size
threshold; results must match the base `NumpyEngine` (statevector semantics) to
floating-point tolerance on the statevector and exactly on deterministic-outcome
counts. Small test states are forced onto the Numba kernel with
``numba_min_size=1`` (the default threshold would route them to the tensordot
reference). Both dispatch branches are exercised: below-threshold falls back to
`_apply_sv`, at/above-threshold uses `apply_matrix_inplace_parallel`.
"""

import numpy as np
import pytest

pytest.importorskip("numba")  # optional dependency; skip suite if absent

import fatqat as fq
from fatqat.backends.engine_contract import _StateVectorResultRequest
from fatqat.backends.numpy_engine import NumpyEngine
from fatqat.backends.statevector_backend import StateVectorBackend
from fatqat.backends.statevector_numba import (
    _NUMBA_PARALLEL_MIN_SIZE,
    NumbaEngine,
    StateVectorBackendNumba,
)
from fatqat.backends.steps import ApplyMatrixStep
from fatqat.implementation.matrices import shift_matrix


def _base_engine():
    return NumpyEngine(state_semantics="statevector")


def _numba_engine(numba_min_size=1):
    return NumbaEngine(state_semantics="statevector", numba_min_size=numba_min_size)


def _lower(program):
    backend = StateVectorBackend()
    layout = backend.resolve_layout(program)
    plan, _facts = backend._lower(program, layout)
    return plan, layout


def _run_engine(engine, program, request, shots=1, seed=None):
    plan, layout = _lower(program)
    engine.initialize(layout.system_dims, layout.n_clbits)
    return engine.run(plan, shots, seed, request)


def _single_h():
    p = fq.Program(1)
    p.add(fq.ops.H, 0)
    return p


def _ghz(n):
    p = fq.Program(n)
    p.add(fq.ops.H, 0)
    for q in range(n - 1):
        p.add(fq.ops.CX, (q, q + 1))
    return p


def _mixed_gates():
    p = fq.Program(4)
    p.add(fq.ops.H, 0)
    p.add(fq.ops.RX(0.7), 1)
    p.add(fq.ops.CX, (0, 1))
    p.add(fq.ops.T, 2)
    p.add(fq.ops.Swap, (1, 2))
    p.add(fq.ops.RZ(1.3), 3)
    p.add(fq.ops.CCX, (0, 1, 3))
    return p


@pytest.mark.parametrize("factory", [_single_h, lambda: _ghz(3), lambda: _ghz(5), _mixed_gates])
def test_fast_path_statevector_matches_base(factory):
    program = factory()
    request = _StateVectorResultRequest(counts=False, statevector=True)
    base = _run_engine(_base_engine(), program, request)
    numba = _run_engine(_numba_engine(numba_min_size=1), program, request)
    assert np.allclose(base.state, numba.state)


@pytest.mark.parametrize("factory", [lambda: _ghz(4), _mixed_gates])
def test_both_dispatch_branches_match_base(factory):
    # numba_min_size=1 forces the parallel kernel; a threshold above the state
    # size forces the tensordot fallback. Both must reproduce the base engine.
    program = factory()
    request = _StateVectorResultRequest(counts=False, statevector=True)
    base = _run_engine(_base_engine(), program, request)
    for min_size in (1, _NUMBA_PARALLEL_MIN_SIZE + 1):
        numba = _run_engine(
            _numba_engine(numba_min_size=min_size), program, request
        )
        assert np.allclose(base.state, numba.state), f"min_size={min_size}"


def test_qudit_apply_matches_base():
    # Qutrit dims flow initialize -> apply -> kernel; force the Numba branch.
    dims = (3, 3)
    step = ApplyMatrixStep(matrix=shift_matrix(3, 1), target_indices=(0,))
    base = _base_engine()
    base.initialize(dims)
    base.apply(step)
    numba = _numba_engine(numba_min_size=1)
    numba.initialize(dims)
    numba.apply(step)
    assert np.allclose(base.export_state(), numba.export_state())


def test_backend_deterministic_counts_fast_path():
    program = fq.Program(3, 3)
    for q in range(3):
        program.add(fq.ops.X, q)
    for q in range(3):
        program.add_measurement(q, q)
    base = StateVectorBackend().run(program, shots=50, seed=1).result().get_counts()
    numba = (
        StateVectorBackendNumba(numba_min_size=1)
        .run(program, shots=50, seed=1)
        .result()
        .get_counts()
    )
    assert base == numba == {"111": 50}


def test_backend_dynamic_reset_counts_match():
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.Reset, 0)
    base = StateVectorBackend().run(program, shots=16, seed=2).result().get_counts()
    numba = (
        StateVectorBackendNumba(numba_min_size=1)
        .run(program, shots=16, seed=2)
        .result()
        .get_counts()
    )
    assert base == numba == {"1": 16}


def test_backend_dynamic_condition_counts_match():
    # X(0); measure c0; X(1) iff c0==1; measure c1. Outcome is deterministic, so
    # the two engines must agree exactly despite kernel differences.
    program = fq.Program(2, 2)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 1, condition=(program.creg[0][0], 1))
    program.add_measurement(1, 1)
    base = StateVectorBackend().run(program, shots=10, seed=3).result().get_counts()
    numba = (
        StateVectorBackendNumba(numba_min_size=1)
        .run(program, shots=10, seed=3)
        .result()
        .get_counts()
    )
    assert base == numba
    assert sum(numba.values()) == 10


def test_default_threshold_backend_runs_via_tensordot_branch():
    # Default threshold (~2**18) routes a small program to the tensordot branch;
    # it must still be correct end to end.
    program = fq.Program(2, 2)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)
    counts = StateVectorBackendNumba().run(program, shots=8, seed=0).result().get_counts()
    assert counts == {"01": 8}  # little-endian: highest clbit first, q0=1


def test_stochastic_fast_path_numba_produces_valid_distribution():
    # Bell state measured: sampling runs through the Numba apply on the fast
    # path. Cross-engine counts need not match bit-for-bit, but the numba run
    # must yield a valid, correctly-supported distribution.
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)
    counts = (
        StateVectorBackendNumba(numba_min_size=1)
        .run(program, shots=200, seed=7)
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 200
    assert set(counts) <= {"00", "11"}  # Bell correlations
    assert set(counts) == {"00", "11"}  # both branches sampled at 200 shots
