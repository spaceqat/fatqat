"""Tests LOSimulator's fixed-photon method against its statevector method."""

import itertools
import math

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.implementation import MatrixImplementationMap
from fatqat.noise import Depolarizing
from fatqat.simulator import (
    BasisState,
    LOSimulator,
    Simulator,
    fixed_photon_index,
    fixed_photon_occupation,
)

_STATE_ONLY = {"shots": 0, "result_config": {"counts": False, "final_state": True}}


def _mesh(num_modes, seed=0):
    """Return a program of random beam splitters and phases on every pair."""
    rng = np.random.default_rng(seed)
    modes = fq.ModeRegister(num_modes)
    counts = fq.CountRegister(num_modes)
    program = fq.Program([modes], [counts])
    for layer in range(num_modes):
        for first in range(layer % 2, num_modes - 1, 2):
            program.add(
                ops.BeamSplitter(rng.uniform(0, 1.5), rng.uniform(0, 2 * math.pi)),
                (modes[first], modes[first + 1]),
            )
            program.add(ops.PhaseShifter(rng.uniform(0, 2 * math.pi)), modes[first])
    return program, modes, counts


def _measure_all(program, modes, counts):
    program.measure(
        tuple(modes[i] for i in range(modes.size)),
        tuple(counts[i] for i in range(counts.size)),
    )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_fixed_photon_state_is_the_statevector_in_its_sector(runtime):
    program, modes, _counts = _mesh(4)
    # Targets apart and in descending order, so a mode lies between them.
    program.add(ops.BeamSplitter(0.7, 0.2), (modes[3], modes[1]))
    start = BasisState({modes[0]: 1, modes[1]: 2})

    fixed = LOSimulator("fixed_photon_state", runtime=runtime).run(
        program, initial_state=start, **_STATE_ONLY
    )
    fixed = fixed.result()
    statevector = (
        LOSimulator("SV", runtime="numpy")
        .run(program, initial_state=start, **_STATE_ONLY)
        .result()
        .get_statevector()
    )

    assert fixed.metadata["num_photons"] == 3
    amplitudes = fixed.get_fixed_photon_state()
    occupations = fixed_photon_occupation(np.arange(len(amplitudes)), 4, 3)
    # Every occupation of 3 photons in 4 modes, ascending like the statevector.
    assert len(amplitudes) == math.comb(6, 3)
    assert (occupations.sum(axis=1) == 3).all()
    rows = np.ravel_multi_index(occupations.T, (4,) * 4)
    assert (np.diff(rows) > 0).all()
    np.testing.assert_allclose(amplitudes, statevector[rows], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.delete(statevector, rows), 0, rtol=0, atol=1e-12)
    assert fixed.metadata["method"] == "fixed_photon_state"
    assert fixed.metadata["runtime"] == runtime
    with pytest.raises(ResultFieldUnavailableError):
        fixed.get_statevector()


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_fixed_photon_counts_match_statevector_with_feedforward(runtime):
    program, modes, counts = _mesh(3)
    # Hong-Ou-Mandel on modes 0 and 1 leaves 0 or 2 photons in mode 0; the
    # mixing that follows depends on the measured count.
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    program.measure(modes[0], counts[0])
    program.add(
        ops.BeamSplitter(math.pi / 3), (modes[1], modes[2]), condition=((counts[0], 2),)
    )
    _measure_all(program, modes, counts)
    start = BasisState({modes[0]: 1, modes[1]: 1, modes[2]: 1})
    config = {"seed": 3, "shot_parallelism": "serial"}
    options = {"shots": 300, "initial_state": start, "simulation_config": config}

    fixed = LOSimulator("fixed_photon_state", runtime=runtime).run(program, **options)
    dense = LOSimulator("SV", runtime=runtime).run(program, **options)

    # Same probabilities in the same order draw the same outcomes.
    assert fixed.result().get_counts() == dense.result().get_counts()
    assert all(sum(map(int, key)) == 3 for key in fixed.result().get_counts())


def test_fixed_photon_runs_where_a_statevector_cannot_be_stored():
    program, modes, _counts = _mesh(12)
    program.add(ops.BeamSplitter(0.7, 0.2), (modes[0], modes[7]))
    start = BasisState({modes[index]: 1 for index in range(6)})

    result = LOSimulator("fixed_photon_state").run(
        program, initial_state=start, **_STATE_ONLY
    )

    # 12 modes with a cutoff of 6 would need 7**12 (about 1.4e10) amplitudes.
    assert result.result().metadata["runtime"] == "numba"  # the default
    state = result.result().get_fixed_photon_state()
    assert len(state) == math.comb(17, 6)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    numpy = LOSimulator("fixed_photon_state", runtime="numpy").run(
        program, initial_state=start, **_STATE_ONLY
    )
    # Threads split the rows; each group is still updated once.
    threaded = LOSimulator("fixed_photon_state").run(
        program,
        initial_state=start,
        simulation_config={"kernel_parallelism": "threads", "max_workers": 3},
        **_STATE_ONLY,
    )
    np.testing.assert_allclose(
        threaded.result().get_fixed_photon_state(), state, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        numpy.result().get_fixed_photon_state(), state, rtol=0, atol=1e-12
    )


def test_fixed_photon_sweep_matches_explicit_runs():
    phi = fq.Parameter("phi")
    modes = fq.ModeRegister(2)
    program = fq.Program([modes])
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    program.add(ops.PhaseShifter(phi), modes[1])
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    values = np.linspace(0.0, math.pi, 3)
    backend = LOSimulator("fixed_photon_state")
    options = {"initial_state": BasisState({modes[0]: 2}), **_STATE_ONLY}

    swept = backend.run_sweep(program, {phi: values}, **options).result()

    for result, value in zip(swept, values, strict=True):
        bound = program.assign_parameters({phi: value})
        expected = backend.run(bound, **options).result().get_fixed_photon_state()
        np.testing.assert_allclose(
            result.get_fixed_photon_state(), expected, rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_fixed_photon_rejects_a_gate_that_changes_the_photon_number(runtime):
    modes = fq.ModeRegister(1)
    program = fq.Program([modes])
    program.add(ops.PhaseShifter(0.1), modes[0])
    adds_a_photon = MatrixImplementationMap()
    adds_a_photon.add(ops.PhaseShifter, np.array([[0.0, 1.0], [1.0, 0.0]]))
    backend = LOSimulator(
        "fixed_photon_state", runtime=runtime, implementation_map=adds_a_photon
    )

    with pytest.raises(BackendValidationError, match="conserve the photon number"):
        backend.run(
            program, initial_state=BasisState({modes[0]: 1}), **_STATE_ONLY
        ).result()


def test_fixed_photon_rejects_what_it_cannot_store():
    program, modes, _counts = _mesh(2)
    start = BasisState({modes[0]: 2})
    noise = fq.NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.BeamSplitter)
    with pytest.raises(BackendValidationError, match="non-unitary"):
        LOSimulator("fixed_photon_state", noise=noise).run(
            program, initial_state=start, **_STATE_ONLY
        )

    with pytest.raises(BackendValidationError, match="not a dense initial_state"):
        LOSimulator("fixed_photon_state").run(program, initial_state=np.eye(9)[1])

    # Every mode must be able to hold every photon.
    with pytest.raises(BackendValidationError, match="cutoff of at least the 2"):
        LOSimulator("fixed_photon_state").run(
            program, initial_state=start, simulation_config={"cutoff": 1}
        )

    qubits = fq.Program(1)
    qubits.add(ops.X, 0)
    with pytest.raises(BackendValidationError, match="all ModeRegisters"):
        LOSimulator("fixed_photon_state").run(qubits, **_STATE_ONLY)


def test_fixed_photon_is_a_method_of_losimulator_only():
    with pytest.raises(BackendValidationError, match="unsupported method"):
        Simulator("fixed_photon_state")


@pytest.mark.parametrize(("num_modes", "num_photons"), [(1, 0), (1, 3), (3, 0), (4, 3)])
def test_fixed_photon_rows_list_occupations_lexicographically(num_modes, num_photons):
    occupations = [
        occupation
        for occupation in itertools.product(range(num_photons + 1), repeat=num_modes)
        if sum(occupation) == num_photons
    ]

    assert [fixed_photon_index(occupation) for occupation in occupations] == list(
        range(len(occupations))
    )
    assert [
        fixed_photon_occupation(row, num_modes, num_photons)
        for row in range(len(occupations))
    ] == occupations
    rows = np.arange(len(occupations)).reshape(-1, 1)
    np.testing.assert_array_equal(
        fixed_photon_occupation(rows, num_modes, num_photons),
        np.array(occupations).reshape(-1, 1, num_modes),
    )
    np.testing.assert_array_equal(fixed_photon_index(np.array(occupations)), rows[:, 0])


def test_fixed_photon_index_functions_reject_bad_input():
    with pytest.raises(ValueError, match="non-negative integers"):
        fixed_photon_index((1, -1))
    with pytest.raises(ValueError, match="non-negative integers"):
        fixed_photon_index((0.5, 1))
    with pytest.raises(ValueError, match=r"index must be integers in \[0, 3\)"):
        fixed_photon_occupation(3, num_modes=2, num_photons=2)
    with pytest.raises(ValueError, match="num_modes must be an int >= 1"):
        fixed_photon_occupation(0, num_modes=0, num_photons=2)
