import numpy as np

from fatqat._backends.engine_contract import (
    _StateVectorResultRequest,
)
from fatqat._backends.steps import (
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    ResetStep,
)
from fatqat.simulator._engine.np import NumpySVEngine
from fatqat.simulator._execution_contract import _ExecutionContext, _ExecutionPolicy

_SERIAL = _ExecutionPolicy(
    shot_strategy="none",
    kernel_strategy="serial",
    worker_limit=1,
    fusion=False,
)


def _context(
    *,
    execution_shape="single_pass",
    n_clbits,
    shots,
    request,
    seed=0,
    initial_state=None,
    initial_occupied=None,
):
    return _ExecutionContext(
        execution_shape=execution_shape,
        request=request,
        system_dims=(2,),
        n_clbits=n_clbits,
        shots=shots,
        seed=seed,
        initial_state=initial_state,
        initial_occupied=initial_occupied,
    )


def _run(engine, plan, context, *, deferred_measurements=(), policy=_SERIAL):
    payload = engine.materialize_execution(
        tuple(plan),
        system_dims=context.system_dims,
        n_clbits=context.n_clbits,
        deferred_measurements=tuple(deferred_measurements),
        policy=policy,
    )
    return engine.execute_local(context, payload, policy)


def test_numba_compiled_multi_shot_compatibility():
    from fatqat.simulator._engine import nb

    compatible = (MeasurementStep((0,), (0,)), ResetStep((0,)))
    unsupported = compatible + (LossStep((0,), p=0.1),)
    engine = nb.NumbaSVEngine()

    assert engine.compiled_multi_shot_compatible(compatible) is True
    assert engine.compiled_multi_shot_compatible(unsupported) is False


def test_engine_fast_counts_returns_arrays():
    engine = NumpySVEngine()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [ApplyMatrixStep(x, (0,)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=1,
        shots=4,
        request=_StateVectorResultRequest(counts=True, statevector=False),
    )

    result = _run(engine, plan, context, deferred_measurements=((0, 0),))

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
        request=_StateVectorResultRequest(counts=True, statevector=True),
    )

    result = _run(engine, plan, context, deferred_measurements=((0, 0),))

    measured = int(result.outcome_keys[0, 0])
    assert result.outcome_counts.tolist() == [1]
    assert np.isclose(abs(result.state[measured]), 1.0)
    assert np.count_nonzero(np.abs(result.state) > 1e-12) == 1
