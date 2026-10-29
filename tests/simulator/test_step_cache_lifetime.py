"""Prepared step data follows plan transitions on a reused simulator."""

import gc
import weakref

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep, BuiltinKernelKey
from fatqat.simulator import Simulator


def _array_refs(value):
    if isinstance(value, np.ndarray):
        return [weakref.ref(value)]
    if isinstance(value, tuple):
        return [ref for part in value for ref in _array_refs(part)]
    return []


def _observe_resolutions(monkeypatch, engine, name):
    """Observe ownership without retaining steps, arrays, or cache containers."""
    original = getattr(engine, name)
    observations = []

    def observe(step):
        resolved = original(step)
        source = step.matrix if isinstance(step, ApplyMatrixStep) else step.kraus_ops
        observations.append(
            (weakref.ref(step), _array_refs(source), _array_refs(resolved))
        )
        return resolved

    monkeypatch.setattr(engine, name, observe)
    return observations


def _assert_released(observations):
    gc.collect()
    for step, source, resolved in observations:
        assert step() is None
        assert all(ref() is None for ref in source + resolved)


@pytest.mark.parametrize(
    "method,runtime,noisy,fusion",
    [
        ("statevector", "numba", False, False),
        ("density_matrix", "numba", False, False),
        ("unitary", "numba", False, False),
        ("superop", "numba", False, False),
        ("statevector", "numpy", True, False),
        ("statevector", "numba", True, False),
        ("density_matrix", "numba", True, False),
        ("superop", "numba", True, True),
    ],
)
def test_replaced_plan_releases_steps_and_arrays(
    monkeypatch, method, runtime, noisy, fusion
):
    if runtime == "numba":
        pytest.importorskip("numba")
    noise = fq.NoiseModel()
    if noisy:
        noise.add(fq.noise.Depolarizing(p=0.2), operation=ops.RY)
    backend = Simulator(method, runtime=runtime, noise=noise)
    if method == "statevector" and noisy:
        cache = backend._engine._channel_routes
    elif method in ("statevector", "unitary"):
        cache = backend._engine._matrix_structures
    else:
        cache = backend._engine._superoperators
    observed = _observe_resolutions(monkeypatch, cache, "resolve")
    config = {"seed": 7, "shot_parallelism": "serial", "kernel_parallelism": "serial"}
    options = {"shots": 1, "result_config": {"counts": False, "final_state": True}}
    previous = []
    for angle in (0.1, 0.1, 0.6):
        program = fq.Program(1)
        program.add(ops.RY(angle), 0)
        result = backend.run(
            program, simulation_config={**config, "fusion": fusion}, **options
        ).result()
        reference = (
            Simulator(method, runtime="numpy", noise=noise)
            .run(program, simulation_config=config, **options)
            .result()
        )
        accessor = f"get_{method}"
        assert np.allclose(getattr(result, accessor)(), getattr(reference, accessor)())
        assert observed
        if fusion:
            # Observe the generated channel, not just the original gate.
            assert not isinstance(observed[-1][0](), ApplyMatrixStep)
        _assert_released(previous)
        previous = list(observed)

    backend.run(fq.Program(1), **options).result()
    _assert_released(observed)


@pytest.mark.parametrize(
    "method", ["statevector", "density_matrix", "unitary", "superop"]
)
def test_sweep_keeps_fixed_resolutions_and_releases_replaced_rows(monkeypatch, method):
    pytest.importorskip("numba")
    backend = Simulator(method, runtime="numba")
    cache = (
        backend._engine._matrix_structures
        if method in ("statevector", "unitary")
        else backend._engine._superoperators
    )
    observed = _observe_resolutions(monkeypatch, cache, "resolve")
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.RY(theta), 0)
    angles = [0.1, 0.1, 0.6]
    swept = backend.run_sweep(program, {theta: angles}).result()

    # Identify operations, not the resolution order or call count of a
    # particular execution path.
    gc.collect()
    alive = [item for item in observed if item[0]() is not None]
    fixed = [item for item in alive if item[0]().kernel_key == BuiltinKernelKey.H]
    parameterized = {
        id(item[0]()) for item in alive if item[0]().kernel_key == BuiltinKernelKey.RY
    }
    assert fixed
    for item in fixed:
        assert item[0]() is fixed[0][0]()
        # Reusing arrays avoids rebuilding fixed numerical data per entry.
        assert all(
            before() is not None and before() is after()
            for before, after in zip(fixed[0][2], item[2], strict=True)
        )
    # Only the last parametric step may stay cached for the next plan.
    assert len(parameterized) <= 1
    _assert_released([item for item in observed if item[0]() is None])
    for angle, result in zip(angles, swept, strict=True):
        reference = (
            Simulator(method, runtime="numpy")
            .run(program.assign_parameters({theta: angle}))
            .result()
        )
        accessor = f"get_{method}"
        assert np.allclose(getattr(result, accessor)(), getattr(reference, accessor)())


def test_superop_retains_only_active_reset_resolutions(monkeypatch):
    pytest.importorskip("numba")
    backend = Simulator("superop", runtime="numba")
    observed = _observe_resolutions(
        monkeypatch, backend._engine._superoperators, "resolve"
    )
    program = fq.Program(1)
    program.add(ops.Reset, 0)
    for _ in range(2):
        result = backend.run(program).result().get_superop()
        assert np.array_equal(
            result, np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        )
    assert observed[0][0]() is observed[1][0]()
    assert all(
        before() is not None and before() is after()
        for before, after in zip(observed[0][2], observed[1][2], strict=True)
    )
    backend.run(fq.Program(1)).result()
    gc.collect()
    # The structural reset-channel cache still owns its step, but the unused
    # prepared super-operator and its classification arrays can be released.
    assert observed[0][0]() is not None
    assert all(ref() is None for ref in observed[0][2])


def test_next_plan_releases_cache_data_after_execution_failure(monkeypatch):
    pytest.importorskip("numba")
    backend = Simulator("SV", runtime="numba")
    cache = backend._engine._matrix_structures
    observed = _observe_resolutions(monkeypatch, cache, "resolve")
    resolve = cache.resolve

    def fail_after_resolving(step):
        resolve(step)
        raise RuntimeError("execution failed after resolving a step")

    program = fq.Program(1)
    program.add(ops.RY(0.1), 0)
    with monkeypatch.context() as patch:
        patch.setattr(cache, "resolve", fail_after_resolving)
        job = backend.run(program, simulation_config={"error_handling": "capture"})
        with pytest.raises(RuntimeError, match="execution failed"):
            job.result()
        del job

    result = backend.run(fq.Program(1)).result().get_statevector()
    assert np.array_equal(result, [1, 0])
    _assert_released(observed)


def test_reused_engine_layouts_follow_system_dims():
    pytest.importorskip("numba")
    backend = Simulator("SV", runtime="numba")
    options = {"shots": 0, "result_config": {"counts": False, "final_state": True}}
    for width, kernel in ((2, "serial"), (3, "serial"), (3, "threads"), (2, "threads")):
        program = fq.Program(width)
        for qubit in range(width):
            program.add(ops.H, qubit)
        program.add(ops.CX, (0, width - 1))
        program.add(ops.RY(0.3), width - 1)
        result = backend.run(
            program,
            simulation_config={"kernel_parallelism": kernel},
            **options,
        ).result()
        reference = Simulator("SV", runtime="numpy").run(program, **options).result()
        assert np.allclose(result.get_statevector(), reference.get_statevector())
