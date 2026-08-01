"""Tests density-matrix simulator initialization, evolution, measurement, and reset."""

import numpy as np
import pytest

from fatqat.simulator.engine.np import NumpyDMEngine, NumpySVEngine
from fatqat._backends.steps import ApplyMatrixStep, MeasurementStep, ResetStep

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)


def _engine(n):
    eng = NumpyDMEngine()
    eng.initialize((2,) * n)
    return eng


def _is_dynamic(plan):
    """Density-matrix dynamic classification (reset is a deterministic channel)."""
    return NumpyDMEngine()._analyze_plan(plan)[0]


def _pure(statevector):
    """Return |psi><psi| for a statevector."""
    psi = np.asarray(statevector, dtype=complex)
    return np.outer(psi, psi.conj())


def test_initialize_zero_state():
    eng = _engine(2)
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1.0
    assert np.allclose(eng.export_state(), expected)


def test_x_on_qubit0_of_two():
    # little-endian: qubit0 is bit0, so |00><00| -> |01><01| at index 1
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    assert np.allclose(eng.export_state(), _pure([0, 1, 0, 0]))


def test_bell_state_h_then_cx():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_CX, target_indices=(0, 1)))
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(eng.export_state(), _pure(bell))


def test_apply_matches_statevector_semantics_on_pure_states():
    # rho evolution must track |psi><psi| through a non-trivial circuit,
    # including a complex-phase gate and a reversed-order two-qubit target.
    steps = [
        ApplyMatrixStep(matrix=_H, target_indices=(0,)),
        ApplyMatrixStep(matrix=_S, target_indices=(0,)),
        ApplyMatrixStep(matrix=_CX, target_indices=(1, 0)),
        ApplyMatrixStep(matrix=_H, target_indices=(2,)),
        ApplyMatrixStep(matrix=_CX, target_indices=(2, 1)),
    ]
    sv = NumpySVEngine()
    sv.initialize((2, 2, 2))
    dm = _engine(3)
    for step in steps:
        sv.apply(step)
        dm.apply(step)
    assert np.allclose(dm.export_state(), _pure(sv.export_state()))


def test_apply_matrix_rho_qutrit():
    shift = np.roll(np.eye(3, dtype=complex), 1, axis=0)  # |k> -> |k+1 mod 3>
    eng = NumpyDMEngine()
    eng.initialize((3,))  # rho = |0><0|
    eng.apply(ApplyMatrixStep(matrix=shift, target_indices=(0,)))
    expected = np.zeros((3, 3), dtype=complex)
    expected[1, 1] = 1.0
    assert np.allclose(eng.export_state(), expected)


def test_probabilities_are_diagonal():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_CX, target_indices=(0, 1)))
    assert np.allclose(eng.probabilities(), [0.5, 0, 0, 0.5])


def test_measure_subsystems_collapses_bell_pair_consistently():
    rng = np.random.default_rng(7)
    for _ in range(5):
        eng = _engine(2)
        eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
        eng.apply(ApplyMatrixStep(matrix=_CX, target_indices=(0, 1)))
        b0, b1 = eng.measure_subsystems((0, 1), rng)
        assert b0 == b1
        idx = b0 + 2 * b1
        expected = np.zeros((4, 4), dtype=complex)
        expected[idx, idx] = 1.0
        assert np.allclose(eng.export_state(), expected)


def test_reset_is_deterministic_and_consumes_no_rng():
    # Reset on a density matrix is a channel, not a sampled branch: two
    # engines with different rng streams must agree exactly.
    states = []
    for seed in (0, 1):
        eng = _engine(1)
        eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
        eng.reset_subsystems((0,), np.random.default_rng(seed))
        states.append(eng.export_state())
    assert np.array_equal(states[0], states[1])
    assert np.allclose(states[0], _pure([1, 0]))


def test_reset_of_entangled_qubit_yields_mixed_state():
    # Reset qubit 0 of a Bell pair: rho' = |0><0| (x) I/2, purity 1/2.
    # This is the representational win over the statevector backend.
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_CX, target_indices=(0, 1)))
    eng.reset_subsystems((0,))
    rho = eng.export_state()
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 0.5  # |00>
    expected[2, 2] = 0.5  # |0>_q0 |1>_q1
    assert np.allclose(rho, expected)
    assert np.isclose(np.real(np.trace(rho @ rho)), 0.5)


def test_reset_grouped_multiple_subsystems():
    eng = _engine(2)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(1,)))
    eng.reset_subsystems((0, 1))
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1.0
    assert np.allclose(eng.export_state(), expected)


def test_reset_requires_at_least_one_index():
    eng = _engine(1)
    with pytest.raises(AssertionError):
        eng.reset_subsystems(())


def test_uninitialized_engine_raises():
    eng = NumpyDMEngine()
    with pytest.raises(RuntimeError, match="initialize"):
        eng.export_state()


def test_plan_with_unconditional_reset_is_not_dynamic():
    # Statevector must go per-shot for reset; the density-matrix channel
    # reset keeps the fast path.
    plan = [
        ApplyMatrixStep(matrix=_H, target_indices=(0,)),
        ResetStep(reset_indices=(0,)),
    ]
    assert _is_dynamic(plan) is False


def test_plan_with_conditioned_reset_is_dynamic():
    plan = [ResetStep(reset_indices=(0,), condition=((0, 1),))]
    assert _is_dynamic(plan) is True


def test_plan_with_reset_on_measured_subsystem_is_dynamic():
    plan = [
        MeasurementStep(measured_indices=(0,), classical_indices=(0,)),
        ResetStep(reset_indices=(0,)),
    ]
    assert _is_dynamic(plan) is True


def test_plan_with_gate_on_measured_subsystem_is_dynamic():
    plan = [
        MeasurementStep(measured_indices=(0,), classical_indices=(0,)),
        ApplyMatrixStep(matrix=_X, target_indices=(0,)),
    ]
    assert _is_dynamic(plan) is True
