import numpy as np
import pytest

from fatqat.program import Program
from fatqat.simulator import Simulator
import fatqat.operations as ops
from fatqat.simulator._engine.state import EvolutionState
from fatqat.simulator._engine.numerics.numpy_kernels._measurement import (
    _statevector_probabilities,
    _project_statevector,
    _sample_basis_indices,
)
from fatqat.simulator._engine.workflows.python_workflow import (
    _MeasurementKernels,
    _measure_subsystems,
)


@pytest.fixture(name="sv_reset", params=["numpy", "numba"])
def fixture_sv_reset(request):
    if request.param == "numba":
        pytest.importorskip("numba")
        from fatqat.simulator._engine.numerics.numba_kernels._measurement import (
            _reset_statevector,
        )
    else:
        from fatqat.simulator._engine.numerics.numpy_kernels._reset import (
            _reset_statevector,
        )
    return _reset_statevector


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_repeated_measurement_reports_the_same_outcome(runtime):
    program = Program(1, 2)
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.measure(0, 1)
    counts = (
        Simulator("SV", runtime=runtime)
        .run(
            program,
            shots=32,
            execution_config={"seed": 0, "shot_parallelism": "serial"},
        )
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 32
    assert set(counts) <= {"00", "11"}


def test_reset_qubit_on_entangled_pair_conditions_the_partner(sv_reset):
    # Resetting q0 of a Bell pair conditions q1 on the sampled branch.
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    outcomes = []
    for seed in range(200):
        state = sv_reset(bell.copy(), (0,), (2, 2), np.random.default_rng(seed))
        nonzero = np.flatnonzero(np.abs(state) > 1e-12)
        assert nonzero.size == 1 and nonzero[0] in (0, 2)
        assert np.isclose(np.linalg.norm(state), 1)
        outcomes.append(int(nonzero[0] == 2))
    assert 0.35 < sum(outcomes) / len(outcomes) < 0.65


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_measurement_preserves_requested_target_order(runtime):
    program = Program(3, 2)
    program.add(ops.X, 2)
    program.measure((2, 0), (0, 1))
    counts = (
        Simulator("SV", runtime=runtime)
        .run(
            program,
            shots=1,
            execution_config={"seed": 0},
        )
        .result()
        .get_counts_as_tuples()
    )
    assert counts == {(1, 0): 1}


def test_measure_subsystems_consumes_one_rng_draw_for_grouped_event():
    # Seed compatibility deliberately constrains grouped measurement to one draw.
    state = EvolutionState(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
    rng_grouped = np.random.default_rng(11)
    kernels = _MeasurementKernels(
        _statevector_probabilities, _project_statevector, _sample_basis_indices
    )
    _measure_subsystems(state, kernels, (2, 2), (0, 1), rng_grouped)
    rng_one_draw = np.random.default_rng(11)
    rng_one_draw.choice(4, p=np.array([0.5, 0.0, 0.0, 0.5]))
    assert rng_grouped.random() == rng_one_draw.random()


def test_reset_subsystems_resets_all_targets_with_one_grouped_collapse(sv_reset):
    state = np.zeros(12, dtype=complex)
    state[[5, 11]] = 1 / np.sqrt(2)  # q0=1, q1=2; q2 remains in superposition.
    rng = np.random.default_rng(0)
    result = sv_reset(state, (1, 0), (2, 3, 2), rng)
    expected = np.zeros(12, dtype=complex)
    expected[[0, 6]] = 1 / np.sqrt(2)
    assert np.allclose(result, expected)
    reference_rng = np.random.default_rng(0)
    reference_rng.random()
    assert rng.random() == reference_rng.random()
