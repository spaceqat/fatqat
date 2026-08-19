"""Tests for Simulator parameter sweeps and shared unbound guards."""

import inspect
from fractions import Fraction

import numpy as np
import pytest

import fatqat as fq
from fatqat.job import Job
from fatqat.resource_layout import ResourceLayout
from fatqat.result import Result
from fatqat.simulator import Simulator


def _rotation_template():
    angles = fq.ParameterVector("angles", 2)
    bias = fq.Parameter("bias")
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)
    program.add(fq.ops.RZ(bias), 0)
    return program, angles, bias


def test_scalar_parameter_sweep_returns_ordered_results():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)
    values = np.array([0.0, np.pi / 2, np.pi])

    results = (
        Simulator("SV")
        .run_sweep(
            program,
            {theta: values},
            shots=0,
            result_config={"counts": False, "final_state": True},
        )
        .result()
    )

    assert isinstance(results, list)
    assert len(results) == len(values)
    assert np.allclose(results[0].get_statevector(), [1, 0])
    assert np.allclose(np.abs(results[2].get_statevector()), [0, 1])


def test_vector_and_multiple_parameter_batches_match_explicit_loop():
    program, angles, bias = _rotation_template()
    bindings = {
        bias: np.array([0.3, 0.4, 0.5]),
        angles: np.array([[0.1, 0.2], [0.4, 0.6], [0.7, 0.8]]),
    }
    backend = Simulator("SV")
    options = {
        "shots": 0,
        "result_config": {"counts": False, "final_state": True},
    }

    swept = backend.run_sweep(program, bindings, **options).result()
    explicit = [
        backend.run(
            program.assign_parameters(
                {angles: bindings[angles][index], bias: bindings[bias][index]}
            ),
            **options,
        ).result()
        for index in range(3)
    ]

    assert all(
        np.allclose(sweep_result.get_statevector(), explicit_result.get_statevector())
        for sweep_result, explicit_result in zip(swept, explicit, strict=True)
    )


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_small_method_smoke_matches_repeated_runs(method):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RY(theta), 0)
    values = np.array([0.2, 0.9])
    backend = Simulator(method, runtime="numpy")

    swept = backend.run_sweep(
        program,
        {theta: values},
        shots=0,
        result_config={"counts": False, "final_state": True},
    ).result()
    explicit = [
        backend.run(
            program.assign_parameters({theta: value}),
            shots=0,
            result_config={"counts": False, "final_state": True},
        ).result()
        for value in values
    ]

    accessor = (
        Result.get_statevector if method == "statevector" else Result.get_density_matrix
    )
    assert all(
        np.allclose(accessor(left), accessor(right))
        for left, right in zip(swept, explicit, strict=True)
    )


def test_counts_seed_and_all_options_match_manual_repeated_runs(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1, 1)
    program.add(fq.ops.RY(theta), 0)
    program.measure(0, 0)
    layout = ResourceLayout({program.quantum_registers[0][0]: 0})
    simulation_config = {"seed": 17, "parallel_mode": "serial"}
    result_config = {"counts": True, "final_state": False}
    backend = Simulator("SV")
    original_run = backend.run
    forwarded = []

    def record(bound, **kwargs):
        forwarded.append(kwargs)
        return original_run(bound, **kwargs)

    monkeypatch.setattr(backend, "run", record)
    values = np.array([0.2, 1.1])
    swept = backend.run_sweep(
        program,
        {theta: values},
        shots=64,
        resource_layout=layout,
        simulation_config=simulation_config,
        result_config=result_config,
    ).result()
    monkeypatch.setattr(backend, "run", original_run)
    explicit = [
        original_run(
            program.assign_parameters({theta: value}),
            shots=64,
            resource_layout=layout,
            simulation_config=simulation_config,
            result_config=result_config,
        ).result()
        for value in values
    ]

    assert [result.get_counts() for result in swept] == [
        result.get_counts() for result in explicit
    ]
    assert len(forwarded) == 2
    assert all(
        kwargs
        == {
            "shots": 64,
            "resource_layout": layout,
            "simulation_config": simulation_config,
            "result_config": result_config,
        }
        for kwargs in forwarded
    )


def test_batch_validation_finishes_before_row_zero(monkeypatch):
    program, angles, bias = _rotation_template()
    backend = Simulator("SV")
    calls = 0

    def unexpected_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("row execution must not begin")

    monkeypatch.setattr(backend, "run", unexpected_run)
    with pytest.raises(TypeError, match="real scalars"):
        backend.run_sweep(
            program,
            {
                angles: [[0.1, 0.2], [0.3, Fraction(1, 3)]],
                bias: [0.4, 0.5],
            },
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda _p, a, _b: [(a, [[0.1, 0.2]])], TypeError),
        (lambda _p, _a, _b: {"angles": [[0.1, 0.2]]}, TypeError),
        (lambda _p, a, b: {a: [[0.1, 0.2]], b: 0.3}, ValueError),
        (lambda _p, a, b: {a: [[0.1, 0.2]], b: "bad"}, TypeError),
        (lambda _p, a, b: {a: [[0.1, 0.2]], b: b"bad"}, TypeError),
        (lambda _p, a, b: {a: [[0.1, 0.2]], b: {0: 0.3}}, TypeError),
        (lambda _p, a, b: {a: [[0.1, 0.2], [0.3]], b: [0.4, 0.5]}, ValueError),
        (lambda _p, a, b: {a: [0.1, 0.2], b: [0.3]}, ValueError),
        (lambda _p, a, b: {a: [[0.1, 0.2], [0.3, 0.4]], b: [0.3]}, ValueError),
        (lambda _p, a, b: {a: [[0.1], [0.2]], b: [0.3, 0.4]}, ValueError),
        (lambda _p, a, b: {a: [[0.1, 0.2]], b: []}, ValueError),
        (lambda _p, a, _b: {a: [[0.1, 0.2]]}, ValueError),
        (lambda _p, _a, _b: {}, ValueError),
    ],
)
def test_invalid_batches_raise_before_execution(build, error):
    program, angles, bias = _rotation_template()

    with pytest.raises(error):
        Simulator("SV").run_sweep(program, build(program, angles, bias))


def test_parameter_free_and_zero_width_batches_are_rejected():
    with pytest.raises(ValueError, match="parameterized program"):
        Simulator().run_sweep(fq.Program(1), {})

    empty = fq.ParameterVector("empty", 0)
    with pytest.raises(ValueError, match="zero-length"):
        Simulator().run_sweep(fq.Program(1), {empty: np.empty((2, 0))})


def test_direct_inner_run_failure_propagates_without_returning_job(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)
    backend = Simulator()

    def fail_on_second(bound, **_kwargs):
        if bound.operations[0].operation.theta == 0.2:
            raise RuntimeError("direct row failure")
        return Job.done(Result(metadata={"row": "first"}))

    monkeypatch.setattr(backend, "run", fail_on_second)
    with pytest.raises(RuntimeError, match="direct row failure"):
        backend.run_sweep(program, {theta: [0.1, 0.2]})


def test_failed_point_job_produces_failed_outer_job_without_partial_list(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)
    backend = Simulator()
    error = KeyboardInterrupt("point failed")

    def fail_on_second(bound, **_kwargs):
        if bound.operations[0].operation.theta == 0.2:
            return Job.failed(error)
        return Job.done(Result(metadata={"row": "first"}))

    monkeypatch.setattr(backend, "run", fail_on_second)
    outer = backend.run_sweep(program, {theta: [0.1, 0.2]})

    with pytest.raises(KeyboardInterrupt, match="point failed") as caught:
        outer.result()
    assert caught.value is error


def test_ordinary_simulator_rejects_unbound_before_preparation(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)
    backend = Simulator()

    monkeypatch.setattr(
        backend,
        "_prepare_program",
        lambda *_args, **_kwargs: pytest.fail("preparation must not run"),
    )
    with pytest.raises(
        fq.errors.BackendValidationError,
        match="program has unbound parameters: theta",
    ):
        backend.run(program)


def test_run_sweep_signature_mirrors_run_options():
    run_parameters = inspect.signature(Simulator.run).parameters
    sweep_parameters = inspect.signature(Simulator.run_sweep).parameters

    assert tuple(sweep_parameters)[3:] == tuple(run_parameters)[2:]
