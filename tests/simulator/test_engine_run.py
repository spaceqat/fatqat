from dataclasses import replace

import numpy as np
import pytest

from fatqat import Program
from fatqat.simulator import Simulator, EvolutionState, ClassicalState
from fatqat.simulator._execution_contract import (
    _ResultRequest,
    _SimulationConfig,
    _RunContext,
)
from fatqat._backends.steps import (
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
)
from fatqat.simulator._engine.engines.numpy_engine import (
    NumpySVEngine,
    NumpyUnitaryEngine,
)
from fatqat.errors import BackendValidationError
from fatqat.simulator._engine._execution_policy import _ExecutionPolicy

_SERIAL = _ExecutionPolicy(
    execution_path="single_pass",
    shot_strategy="none",
    kernel_strategy="serial",
    worker_limit=1,
    fusion=False,
)


def _context(
    *,
    n_clbits,
    shots,
    request,
    seed=0,
):
    return _RunContext(
        request=request,
        system_dims=(2,),
        n_clbits=n_clbits,
        shots=shots,
        seed=seed,
    )


def _run(
    engine,
    plan,
    context,
    *,
    policy=_SERIAL,
    initial_state=None,
):
    payload = engine.prepare_payload(
        tuple(plan),
        system_dims=context.system_dims,
        n_clbits=context.n_clbits,
        policy=policy,
    )
    return engine.execute_payload(
        context,
        payload,
        policy,
        initial_state=(
            initial_state if initial_state is not None else EvolutionState()
        ),
    )


@pytest.mark.parametrize("density_matrix", [False, True])
def test_execution_survives_layout_changes(density_matrix):
    pytest.importorskip("numba")
    from fatqat.simulator._engine.engines.numba_engine import (
        NumbaSVEngine,
        NumbaDMEngine,
    )

    engine = NumbaDMEngine() if density_matrix else NumbaSVEngine()
    # Flip the first local target; target order and mixed dimensions determine
    # its position in the full state. Reuse layouts, then change each input.
    matrix = np.kron(np.array([[0, 1], [1, 0]], dtype=complex), np.eye(2))
    threaded = replace(_SERIAL, kernel_strategy="threads", worker_limit=2)
    executions = []
    for dims, targets, policy in (
        ((2, 2), (0, 1), _SERIAL),
        ((2, 2), (0, 1), _SERIAL),
        ((2, 2), (1, 0), _SERIAL),
        ((2, 3, 2), (0, 2), _SERIAL),
        ((2, 3, 2), (0, 2), threaded),
    ):
        context = _RunContext(
            request=_ResultRequest(counts=False, final_state=True),
            system_dims=dims,
            n_clbits=0,
            shots=1,
            seed=0,
        )
        payload = engine.prepare_payload(
            (ApplyMatrixStep(matrix, targets),),
            system_dims=dims,
            n_clbits=0,
            policy=policy,
        )
        expected = np.zeros(np.prod(dims), dtype=complex)
        expected[int(np.prod(dims[: targets[0]]))] = 1
        if density_matrix:
            expected = np.outer(expected, expected.conj())
        executions.append((context, payload, policy, expected))

    # Old payloads must remain usable after both cache reuse and pruning.
    for context, payload, policy, expected in reversed(executions):
        result = engine.execute_payload(
            context, payload, policy, initial_state=EvolutionState()
        )
        np.testing.assert_allclose(result.state, expected)


@pytest.mark.parametrize("with_loss", [False, True])
def test_numba_trajectory_execution_handles_supported_and_fallback_plans(with_loss):
    from fatqat.simulator._engine.engines.numba_engine import NumbaSVEngine

    plan = (ResetStep((0,)),)
    if with_loss:
        plan += (LossStep((0,), p=1.0),)
    plan += (MeasurementStep((0,), (0,)),)
    engine = NumbaSVEngine()
    initial = EvolutionState()
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    (result,) = engine.execute(
        plan,
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="serial", kernel_parallelism="serial"
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[2 if with_loss else 0]]
    assert result.outcome_counts.tolist() == [4]


def test_engine_fast_counts_returns_arrays():
    engine = NumpySVEngine()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [ApplyMatrixStep(x, (0,)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )

    result = _run(engine, plan, context)

    assert result.state is None
    assert result.outcome_keys.tolist() == [[1]]
    assert result.outcome_counts.tolist() == [4]


def test_engine_fast_counts_and_state_share_collapse_event():
    engine = NumpySVEngine()
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    plan = [ApplyMatrixStep(h, (0,)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=1,
        shots=1,
        seed=2026,
        request=_ResultRequest(counts=True, final_state=True),
    )

    result = _run(engine, plan, context)

    measured = int(result.outcome_keys[0, 0])
    assert result.outcome_counts.tolist() == [1]
    assert np.isclose(abs(result.state[measured]), 1.0)
    assert np.count_nonzero(np.abs(result.state) > 1e-12) == 1


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize(
    ("occupied", "reported"),
    [(None, 1), (set(), 0), ({0}, 1)],
)
def test_occupancy_loss_reload_and_feedback_start_fresh(runtime, occupied, reported):
    if runtime == "numba":
        from fatqat.simulator._engine.engines.numba_engine import NumbaSVEngine

        engine = NumbaSVEngine()
    else:
        engine = NumpySVEngine()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [
        ApplyMatrixStep(x, (0,)),
        PutStep((0,)),  # Put preserves loaded atoms and initializes empty sites.
        MeasurementStep((0,), (0,)),
        LossStep((0,), p=1.0),
        MeasurementStep((0,), (1,)),
        PutStep((0,)),
        ApplyMatrixStep(x, (0,), condition=((0, 1),)),
        MeasurementStep((0,), (2,)),
    ]
    context = _context(
        n_clbits=3,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    initial = EvolutionState(classical=ClassicalState(occupied=occupied))
    policy = _ExecutionPolicy(
        execution_path="per_shot",
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=False,
    )

    result = _run(engine, plan, context, policy=policy, initial_state=initial)

    assert result.outcome_keys.tolist() == [[reported, 2, reported]]
    assert result.outcome_counts.tolist() == [4]

    # Reusing the engine restores initial occupancy and clears report digits.
    reused = _run(engine, plan[:3], context, policy=policy, initial_state=initial)
    assert reused.outcome_keys.tolist() == [[reported, 0, 0]]
    assert reused.outcome_counts.tolist() == [4]
    assert result.outcome_keys.tolist() == [[reported, 2, reported]]

    # Unused declared registers still report zero without evolving any digits.
    unwritten = _run(
        engine,
        [plan[0], LossStep((0,), p=0.0)],
        context,
        policy=policy,
        initial_state=initial,
    )
    assert unwritten.outcome_keys.tolist() == [[0, 0, 0]]
    assert unwritten.outcome_counts.tolist() == [4]


@pytest.mark.parametrize("mode", ["numpy", "numba", "compiled", "processes"])
def test_initial_register_drives_feedback_and_is_fresh_per_shot(mode):
    if mode == "numpy":
        engine = NumpySVEngine()
    else:
        from fatqat.simulator._engine.engines.numba_engine import NumbaSVEngine

        engine = NumbaSVEngine()
    initial = EvolutionState(
        quantum=np.array([0, 1], dtype=complex),
        classical=ClassicalState(clbits=[1, 2]),
    )
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [ApplyMatrixStep(x, (0,), condition=((0, 1),)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=2,
        shots=8,
        request=_ResultRequest(counts=True, final_state=False),
    )
    (result,) = engine.execute(
        tuple(plan),
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="processes" if mode == "processes" else "serial",
            kernel_parallelism="serial" if mode != "numba" else "auto",
            max_workers=2 if mode == "processes" else None,
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[0, 2]]
    assert result.outcome_counts.tolist() == [8]
    assert initial.classical.clbits == [1, 2]
    np.testing.assert_array_equal(initial.quantum, [0, 1])


@pytest.mark.parametrize("measure", [False, True])
def test_fast_counts_preserve_unwritten_initial_digits(measure):
    initial = EvolutionState(classical=ClassicalState(clbits=[1, 2]))
    context = _context(
        n_clbits=2,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    plan = [MeasurementStep((0,), (0,))] if measure else []

    result = _run(
        NumpySVEngine(),
        plan,
        context,
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[0 if measure else 1, 2]]
    assert result.outcome_counts.tolist() == [4]


@pytest.mark.parametrize("digits", [[-1], [1.5], [True], [2**63]])
def test_initial_register_rejects_invalid_digits(digits):
    with pytest.raises(BackendValidationError, match="non-negative int64"):
        Simulator().run(
            Program(1, 1),
            initial_state=EvolutionState(classical=ClassicalState(clbits=digits)),
        )


@pytest.mark.parametrize("digits", [[], [1, 2]])
def test_initial_register_requires_matching_width(digits):
    initial = EvolutionState(classical=ClassicalState(clbits=digits))
    with pytest.raises(BackendValidationError, match="expected 1"):
        Simulator().run(Program(1, 1), initial_state=initial)


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_methods_reject_initial_evolution_state(method):
    initial = EvolutionState()
    with pytest.raises(BackendValidationError, match="not meaningful"):
        Simulator(method=method).run(Program(1, 1), initial_state=initial)


def test_operator_engine_accepts_absent_classical_components():
    engine = NumpyUnitaryEngine()
    engine.initialize((2,), initial_state=EvolutionState())
    np.testing.assert_array_equal(engine.export_state(), np.eye(2))


def test_numba_initial_occupancy_preserves_erased_measurement():
    from fatqat.simulator._engine.engines.numba_engine import NumbaSVEngine

    engine = NumbaSVEngine()
    plan = (ResetStep((0,)), MeasurementStep((0,), (0,)))
    initial = EvolutionState(classical=ClassicalState(occupied=set()))
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    (result,) = engine.execute(
        plan,
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="serial", kernel_parallelism="serial"
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[2]]
    assert result.outcome_counts.tolist() == [4]
