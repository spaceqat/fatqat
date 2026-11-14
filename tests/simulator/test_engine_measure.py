"""Measurement mathematics and observable sampling behavior."""

import numpy as np
import pytest

from fatqat.program import Program
from fatqat.simulator import Simulator
import fatqat.operations as ops
from fatqat.simulator._engine.numerics.numpy_kernels._measurement import (
    _project_statevector,
    _statevector_probabilities,
)


def test_probabilities():
    state = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert np.allclose(_statevector_probabilities(state), [0.5, 0.5])


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_samples_follow_the_born_distribution(runtime):
    program = Program(1, 1)
    program.add(ops.H, 0)
    program.measure(0, 0)
    counts = (
        Simulator("SV", runtime=runtime)
        .run(program, shots=2000, execution_config={"seed": 42})
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 2000
    assert 0.45 < counts.get("1", 0) / 2000 < 0.55


@pytest.mark.parametrize("outcome", [0, 2])
def test_partial_projection_preserves_compatible_superposition_and_input(outcome):
    state = np.full(4, 0.5, dtype=complex)
    original = state.copy()
    projected = _project_statevector(state, outcome, (1,), (2, 2))
    expected = np.zeros(4, dtype=complex)
    expected[outcome : outcome + 2] = 1 / np.sqrt(2)
    assert np.array_equal(state, original)
    assert np.allclose(projected, expected)
    assert np.isclose(np.linalg.norm(projected), 1.0)


@pytest.mark.parametrize("outcome", range(4))
def test_full_projection_produces_normalized_basis_state(outcome):
    state = np.full(4, 0.5, dtype=complex)
    projected = _project_statevector(state, outcome, (0, 1), (2, 2))
    expected = np.zeros(4, dtype=complex)
    expected[outcome] = 1
    assert np.allclose(projected, expected)
