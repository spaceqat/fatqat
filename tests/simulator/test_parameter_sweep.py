"""Tests for Simulator parameter sweeps and shared unbound guards."""

import inspect
import pickle
import typing
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.backend_utils import _LoweringContext
from fatqat._backends import steps as steps_module
from fatqat._backends.steps import ApplyMatrixStep, ResolvedStep
from fatqat._index_allocation import _ClassicalAllocation
from fatqat._parameter_binding import _discover_parameters
from fatqat.errors import BackendValidationError, MatrixImplementationError
from fatqat.implementation import MatrixImplementationMap
from fatqat.job import Job
from fatqat.operations import Operation
from fatqat.resource_layout import ResourceLayout
from fatqat.result import Result
from fatqat.simulator import Simulator
from fatqat.simulator.planning import _MatrixRecipe, _ParametricPlan


def _rx_matrix(angle):
    half = angle / 2
    return np.array(
        [
            [np.cos(half), -1j * np.sin(half)],
            [-1j * np.sin(half), np.cos(half)],
        ]
    )


def _rotation_template():
    angles = fq.ParameterVector("angles", 2)
    bias = fq.Parameter("bias")
    program = fq.Program(2)
    program.add(ops.RX(angles[0]), 0)
    program.add(ops.RY(angles[1]), 1)
    program.add(ops.RZ(bias), 0)
    return program, angles, bias


def test_scalar_parameter_sweep_returns_ordered_results():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
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
    program.add(ops.RY(theta), 0)
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


def test_counts_seed_and_options_match_manual_repeated_runs():
    theta = fq.Parameter("theta")
    program = fq.Program(1, 1)
    program.add(ops.RY(theta), 0)
    program.measure(0, 0)
    layout = ResourceLayout({program.quantum_registers[0][0]: 0})
    initial_state = [0.0, 1.0]
    simulation_config = {
        "seed": 17,
        "shot_parallelism": "serial",
        "kernel_parallelism": "serial",
    }
    result_config = {"counts": True, "final_state": False}
    backend = Simulator("SV")
    values = np.array([0.2, 1.1])
    swept = backend.run_sweep(
        program,
        {theta: values},
        shots=64,
        resource_layout=layout,
        initial_state=initial_state,
        simulation_config=simulation_config,
        result_config=result_config,
    ).result()
    explicit = [
        backend.run(
            program.assign_parameters({theta: value}),
            shots=64,
            resource_layout=layout,
            initial_state=initial_state,
            simulation_config=simulation_config,
            result_config=result_config,
        ).result()
        for value in values
    ]

    assert [result.get_counts() for result in swept] == [
        result.get_counts() for result in explicit
    ]


def test_batch_validation_finishes_before_row_zero(monkeypatch):
    program, angles, bias = _rotation_template()
    backend = Simulator("SV")
    calls = 0

    def unexpected_prepare(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("preparation must not begin")

    monkeypatch.setattr(backend, "_prepare_program", unexpected_prepare)
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


def test_nonrectangular_batch_reports_parameter_context():
    program, angles, bias = _rotation_template()

    with pytest.raises(ValueError, match="parameter batch values.*rectangular"):
        Simulator("SV").run_sweep(
            program,
            {
                angles: [np.zeros((2, 2)), np.zeros((2, 3))],
                bias: [0.1, 0.2],
            },
        )


def test_parameter_free_and_zero_width_batches_are_rejected():
    with pytest.raises(ValueError, match="parameterized program"):
        Simulator().run_sweep(fq.Program(1), {})

    empty = fq.ParameterVector("empty", 0)
    with pytest.raises(ValueError, match="zero-length"):
        Simulator().run_sweep(fq.Program(1), {empty: np.empty((2, 0))})


def test_replay_rule_failure_raises_directly_without_returning_job():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)

    def explosive_rx(op, targets):
        if op.theta == 0.2:
            raise RuntimeError("direct row failure")
        return _rx_matrix(op.theta)

    impl_map = MatrixImplementationMap()
    impl_map.add(ops.RX, explosive_rx)
    backend = Simulator(implementation_map=impl_map)

    with pytest.raises(MatrixImplementationError, match="direct row failure"):
        backend.run_sweep(program, {theta: [0.1, 0.2]})


def test_failed_point_execution_produces_failed_outer_job_without_partial_list(
    monkeypatch,
):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    backend = Simulator()
    error = RuntimeError("point failed")
    original_execute = backend._execute_plan
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(backend, "_execute_plan", fail_on_second)
    outer = backend.run_sweep(program, {theta: [0.1, 0.2]})

    with pytest.raises(RuntimeError, match="point failed") as caught:
        outer.result()
    assert caught.value is error


def test_point_job_interrupt_propagates_from_sweep(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    backend = Simulator()
    error = KeyboardInterrupt("point interrupted")

    monkeypatch.setattr(
        backend,
        "_execute_plan",
        lambda *_args, **_kwargs: Job(status="ERROR", error=error),
    )

    with pytest.raises(KeyboardInterrupt, match="point interrupted") as caught:
        backend.run_sweep(program, {theta: [0.1]})
    assert caught.value is error


def test_ordinary_simulator_rejects_unbound_before_preparation(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
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


def test_sweep_lowers_once_for_the_whole_batch(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.RX(theta), 0)
    backend = Simulator("SV")
    original_lower = backend._lower
    lower_calls = 0

    def count_lower(*args, **kwargs):
        nonlocal lower_calls
        lower_calls += 1
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(backend, "_lower", count_lower)

    results = backend.run_sweep(
        program,
        {theta: np.linspace(0.0, 1.0, 5)},
        shots=0,
        result_config={"counts": False, "final_state": True},
    ).result()

    assert lower_calls == 1
    assert len(results) == 5


def test_concrete_rules_run_once_and_parametric_rules_run_per_row():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.RX(theta), 0)
    calls = {"h": 0, "rx": 0}

    def h_rule(op, targets):
        calls["h"] += 1
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def rx_rule(op, targets):
        calls["rx"] += 1
        return _rx_matrix(op.theta)

    impl_map = MatrixImplementationMap()
    impl_map.add(ops.H, h_rule)
    impl_map.add(ops.RX, rx_rule)
    backend = Simulator("SV", implementation_map=impl_map)
    values = np.linspace(0.0, 1.0, 5)

    swept = backend.run_sweep(
        program,
        {theta: values},
        shots=0,
        result_config={"counts": False, "final_state": True},
    ).result()

    assert calls == {"h": 1, "rx": len(values)}
    explicit = backend.run(
        program.assign_parameters({theta: values[2]}),
        shots=0,
        result_config={"counts": False, "final_state": True},
    ).result()
    assert np.allclose(swept[2].get_statevector(), explicit.get_statevector())


def test_parametric_program_lowers_to_deferred_picklable_steps():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.RX(theta), 0)
    backend = Simulator("SV")
    layout = backend._resolve_resource_layout(program)
    param_order = _discover_parameters(program._instructions)
    context = _LoweringContext(
        resource_layout=layout,
        engine_allocation=backend._allocate_engine_indices(program, layout),
        classical_allocation=_ClassicalAllocation.from_program(program),
    )

    plan, _facts, _occupied = backend._prepare_parametric_program(
        program, context=context, param_order=param_order
    )

    assert isinstance(plan, _ParametricPlan)
    assert plan.param_order == param_order
    assert [type(step) for step in plan.steps] == [ApplyMatrixStep, _MatrixRecipe]
    deferred = plan.steps[1]
    assert deferred.param_slots == (0,)
    assert deferred.target_dims == (2,)
    assert deferred.target_indices == (0,)
    # Non-parametric steps are shared with the materialized plans, not rebuilt.
    materialized = plan.materialize((0.3,))
    assert materialized[0] is plan.steps[0]
    assert isinstance(materialized[1], ApplyMatrixStep)
    assert np.allclose(materialized[1].matrix, _rx_matrix(0.3))
    # The deferred plan is a plain execution payload: it pickles by value.
    restored = pickle.loads(pickle.dumps(plan))
    assert np.allclose(restored.materialize((0.3,))[1].matrix, _rx_matrix(0.3))


def test_resolved_step_union_holds_only_engine_executable_steps():
    # ResolvedStep promises every member can be handed to an engine as-is; the
    # deferred sweep recipe lives in the simulator's private plan type instead.
    assert _MatrixRecipe not in typing.get_args(ResolvedStep)
    assert not hasattr(steps_module, "ParametricMatrixStep")


def test_replay_validates_realized_matrix_shape():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)

    def wrong_shape(op, targets):
        return np.eye(4)

    impl_map = MatrixImplementationMap()
    impl_map.add(ops.RX, wrong_shape)
    backend = Simulator(implementation_map=impl_map)

    with pytest.raises(BackendValidationError, match="incompatible with target"):
        backend.run_sweep(program, {theta: [0.1]})


def test_replay_revalidates_targets_after_substitution():
    # Binding through Program.assign_parameters rebuilds the applied operation
    # and re-runs validate_targets(); replaying a recipe must not skip it.
    @dataclass(frozen=True)
    class NonNegativeRotation(Operation):
        theta: float | fq.Parameter
        name: ClassVar[str] = "NonNegativeRotation"
        num_subsystems: ClassVar[int] = 1

        def validate_targets(self, targets):
            if not isinstance(self.theta, fq.Parameter) and self.theta < 0:
                raise ValueError("NonNegativeRotation needs a non-negative angle")

    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(NonNegativeRotation(theta), 0)
    impl_map = MatrixImplementationMap()
    impl_map.add(NonNegativeRotation, lambda op, targets: _rx_matrix(op.theta))
    backend = Simulator("SV", implementation_map=impl_map)

    with pytest.raises(ValueError, match="non-negative angle"):
        backend.run_sweep(
            program,
            {theta: [0.5, -0.5]},
            shots=0,
            result_config={"counts": False, "final_state": True},
        )
