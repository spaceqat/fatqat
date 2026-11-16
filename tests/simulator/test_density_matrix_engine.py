"""Tests density-matrix simulator initialization, evolution, measurement, and reset."""

import numpy as np
import pytest

from fatqat.program import Program
from fatqat.simulator import Simulator
import fatqat.operations as ops
from fatqat.simulator._engine.numerics.numpy_kernels._measurement import (
    _density_matrix_probabilities,
)

from fatqat.simulator._engine.numerics.numpy_kernels._reset import _reset_density_matrix
from fatqat.simulator._engine.engines.numpy_engine import NumpyDMEngine, NumpySVEngine
from fatqat._backends.steps import ApplyMatrixStep

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)


def _engine(n):
    eng = NumpyDMEngine()
    eng.initialize((2,) * n)
    return eng


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
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(_density_matrix_probabilities(_pure(bell)), [0.5, 0, 0, 0.5])


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_measurement_counts_agree_with_collapsed_bell_state(runtime):
    program = Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    simulator = Simulator("DM", runtime=runtime)
    for seed in range(5):
        result = simulator.run(
            program,
            shots=1,
            execution_config={"seed": seed},
            result_config={"counts": True, "final_state": True},
        ).result()
        counts = result.get_counts()
        assert counts in ({"00": 1}, {"11": 1})
        index = 0 if "00" in counts else 3
        expected = np.zeros((4, 4), dtype=complex)
        expected[index, index] = 1
        assert np.allclose(result.get_density_matrix(), expected)


def test_reset_is_deterministic_and_consumes_no_rng():
    initial = _pure(np.array([1, 1]) / np.sqrt(2))
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        state = _reset_density_matrix(initial, (0,), (2,), rng)
        assert np.allclose(state, _pure([1, 0]))
        assert rng.random() == np.random.default_rng(seed).random()


def test_reset_grouped_multiple_subsystems():
    rho = _reset_density_matrix(_pure([0, 0, 0, 1]), (0, 1), (2, 2))
    assert np.allclose(rho, _pure([1, 0, 0, 0]))


def test_uninitialized_engine_raises():
    eng = NumpyDMEngine()
    with pytest.raises(RuntimeError, match="initialize"):
        eng.export_state()
