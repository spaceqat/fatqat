"""Tests for Simulator parameter sweeps and shared unbound guards."""

import inspect
import pickle
import typing
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import ClassVar

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.backend_utils import _LoweringContext
from fatqat._backends.steps import (
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    ResolvedStep,
)
from fatqat._index_allocation import _ClassicalAllocation
from fatqat.errors import BackendValidationError, MatrixImplementationError
from fatqat.implementation import MatrixImplementationMap
from fatqat.operations import Operation
from fatqat.resource_layout import ResourceLayout
from fatqat.result import Result
from fatqat.simulator import Simulator
from fatqat.simulator._engine.binding import _bind_parameter_batch


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


def test_sweep_returns_each_rows_final_classical_state():
    theta = fq.Parameter("theta")
    program = fq.Program(1, 1)
    program.add(ops.RX(theta), 0)
    program.measure(0, 0)
    results = (
        Simulator("SV")
        .run_sweep(
            program,
            {theta: [0.0, np.pi]},
            shots=1,
            result_config={"final_classical_state": True},
        )
        .result()
    )
    assert [result.get_classical_state().clbits for result in results] == [
        [0],
        [1],
    ]


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


_STATE_ONLY = {"shots": 0, "result_config": {"counts": False, "final_state": True}}


@pytest.mark.parametrize(
    ("arity", "build", "targets", "dim"),
    [
        (1, lambda p: ops.RX(p[0]), 0, 2),
        (1, lambda p: ops.RY(p[0]), 0, 2),
        (1, lambda p: ops.RZ(p[0]), 0, 2),
        (1, lambda p: ops.Phase(p[0]), 0, 2),
        (2, lambda p: ops.U(p[0], p[1], 0.7), 0, 2),
        (1, lambda p: ops.U1(p[0]), 0, 2),
        (1, lambda p: ops.U2(0.3, p[0]), 0, 2),
        (3, lambda p: ops.U3(p[0], p[1], p[2]), 0, 2),
        (1, lambda p: ops.CPhase(p[0]), (0, 1), 2),
        (1, lambda p: ops.SubspaceRX(p[0], (0, 2)), 0, 3),
        (1, lambda p: ops.SubspaceRY(p[0], (1, 2)), 0, 3),
        (1, lambda p: ops.SubspaceRZ(p[0], (0, 2)), 0, 3),
        (1, lambda p: ops.PhaseShifter(p[0]), 0, 3),
        (2, lambda p: ops.BeamSplitter(p[0], p[1]), (0, 1), 3),
    ],
    ids=[
        "RX",
        "RY",
        "RZ",
        "Phase",
        "U-mixed",
        "U1",
        "U2",
        "U3",
        "CPhase",
        "SubspaceRX",
        "SubspaceRY",
        "SubspaceRZ",
        "PhaseShifter",
        "BeamSplitter",
    ],
)
def test_every_builtin_parametric_gate_matches_explicit_loop(
    arity, build, targets, dim
):
    params = [fq.Parameter(f"p{index}") for index in range(arity)]
    program = fq.Program([fq.QuantumRegister(2, dim=dim)])
    for qudit in range(2):
        program.add(ops.H if dim == 2 else ops.Fourier, qudit)
    program.add(build(params), targets)
    values = np.random.default_rng(11).uniform(-7.0, 7.0, size=(5, arity))
    backend = Simulator("SV")

    swept = backend.run_sweep(
        program,
        {param: values[:, index] for index, param in enumerate(params)},
        **_STATE_ONLY,
    ).result()
    explicit = [
        backend.run(program.assign_parameters(dict(zip(params, row))), **_STATE_ONLY)
        .result()
        .get_statevector()
        for row in values
    ]

    for result, expected in zip(swept, explicit, strict=True):
        np.testing.assert_allclose(
            result.get_statevector(), expected, rtol=0, atol=1e-12
        )


def test_counts_match_explicit_loop():
    theta = fq.Parameter("theta")
    program = fq.Program(1, 1)
    program.add(ops.RY(theta), 0)
    program.measure(0, 0)
    values = np.linspace(0.0, np.pi, 7)
    backend = Simulator("SV")
    options = {"shots": 16, "simulation_config": {"seed": 5}}

    swept = backend.run_sweep(program, {theta: values}, **options).result()
    explicit = [
        backend.run(program.assign_parameters({theta: value}), **options)
        .result()
        .get_counts()
        for value in values
    ]

    assert [result.get_counts() for result in swept] == explicit


def test_batched_builder_failure_raises_matrix_implementation_error():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.U(theta, 10**400, 0), 0)

    # The NumPy runtime binds sweep values through the host batched rules.
    with pytest.raises(MatrixImplementationError, match="implementation for U"):
        Simulator("SV", runtime="numpy").run_sweep(program, {theta: [0.1, 0.2]})


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
    original_assemble = backend._assemble_result
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return original_assemble(*args, **kwargs)

    # Every sweep row's result is assembled separately, so failing the second
    # assembly fails the second row on any execution path.
    monkeypatch.setattr(backend, "_assemble_result", fail_on_second)
    outer = backend.run_sweep(
        program, {theta: [0.1, 0.2]}, simulation_config={"error_handling": "capture"}
    )

    with pytest.raises(RuntimeError, match="point failed") as caught:
        outer.result()
    assert caught.value is error


def test_point_job_interrupt_propagates_from_sweep(monkeypatch):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    backend = Simulator()
    error = KeyboardInterrupt("point interrupted")

    def interrupt(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(backend._engine, "execute", interrupt)

    # Captured error handling covers Exception, never an interrupt.
    with pytest.raises(KeyboardInterrupt, match="point interrupted") as caught:
        backend.run_sweep(
            program, {theta: [0.1]}, simulation_config={"error_handling": "capture"}
        )
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
    context = _LoweringContext(
        resource_layout=layout,
        engine_allocation=backend._allocate_engine_indices(program, layout),
        classical_allocation=_ClassicalAllocation.from_program(program),
    )

    column = np.array([0.3, 0.6])
    plan, _facts = backend._prepare_program(
        program, context=context, parameter_values={theta: column}
    )

    assert [type(step) for step in plan] == [ApplyMatrixStep, ApplyParametricMatrixStep]
    assert plan[1].operation.theta is theta
    assert plan[1].target_indices == (0,)
    ((name, injected),) = plan[1].values
    assert name == "theta" and injected is column
    # Non-parametric steps are shared with the bound plans, not rebuilt.
    first, second = _bind_parameter_batch(plan, (2,))
    assert first[0] is plan[0]
    assert isinstance(first[1], ApplyMatrixStep)
    assert np.allclose(first[1].matrix, _rx_matrix(0.3))
    assert np.allclose(second[1].matrix, _rx_matrix(0.6))
    # The value-injected plan is plain data: it pickles by value on its own.
    restored, _ = _bind_parameter_batch(pickle.loads(pickle.dumps(plan)), (2,))
    assert np.allclose(restored[1].matrix, _rx_matrix(0.3))


def test_engines_reject_unbound_parametric_steps():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    backend = Simulator("SV")
    plan, _facts = backend._prepare_program(
        program, parameter_values={theta: np.array([0.1])}
    )
    template = (replace(plan[0], values=()),)

    assert ApplyParametricMatrixStep in typing.get_args(ResolvedStep)
    with pytest.raises(RuntimeError, match="injected before binding"):
        next(_bind_parameter_batch(template, (2,)))
    with pytest.raises(RuntimeError, match="bound before execution"):
        # The guard runs before the context, policy, or state is read.
        backend._engine._dispatch_by_shot_strategy(template, None, None, None)


@pytest.mark.parametrize(
    "parallelism",
    [{}, {"shot_parallelism": "threads", "max_workers": 2}],
    ids=["auto", "threads"],
)
def test_noisy_counts_sweep_matches_explicit_runs(parallelism):
    # Noise, counts, and repeated shots select per-shot trajectories. Tilt has
    # a custom rule whose matrix structure changes with its angle: diagonal at
    # 0, a permutation at pi, dense otherwise. Every row must match its own run,
    # also when shots run in threads.
    @dataclass(frozen=True)
    class Tilt(Operation):
        theta: float | fq.Parameter
        name: ClassVar[str] = "Tilt"
        num_subsystems: ClassVar[int] = 1

    def tilt(op):
        return _rx_matrix(op.theta)

    theta, phi, tau = fq.Parameter("theta"), fq.Parameter("phi"), fq.Parameter("tau")
    program = fq.Program(2, 2)
    program.add(ops.RY(theta), 0)
    program.add(ops.CX, (0, 1))
    program.add(ops.RZ(phi), 1)
    program.add(Tilt(tau), 0)
    program.measure(0, 0)
    program.measure(1, 1)
    impl_map = fq.implementation.default_matrix_implementation_map()
    impl_map.add(Tilt, tilt)
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
    backend = Simulator("SV", implementation_map=impl_map, noise=noise)
    values = {theta: [0.1, 1.0, 2.0], phi: [0.0, 0.5, 3.0], tau: [0.0, np.pi, 1.2]}
    options = {"shots": 64, "simulation_config": {"seed": 11, **parallelism}}

    swept = backend.run_sweep(program, values, **options).result()

    explicit = [
        backend.run(
            program.assign_parameters(
                {key: column[row] for key, column in values.items()}
            ),
            **options,
        )
        .result()
        .get_counts()
        for row in range(3)
    ]
    assert [result.get_counts() for result in swept] == explicit


def test_custom_rule_receives_each_row_value():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta), 0)
    received = []

    def recording_rx(op):
        received.append(op.theta)
        return _rx_matrix(op.theta)

    impl_map = MatrixImplementationMap()
    impl_map.add(ops.RX, recording_rx)
    backend = Simulator("SV", implementation_map=impl_map)

    results = backend.run_sweep(program, {theta: [0, 0.5, 1.5]}, **_STATE_ONLY)

    assert received == [0, 0.5, 1.5]
    for result, value in zip(results.result(), received, strict=True):
        expected = _rx_matrix(value) @ np.array([1, 0])
        assert np.allclose(result.get_statevector(), expected)


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


def test_compiled_sweep_formulas_match_numpy_runtime():
    # Every built-in parametric gate, including constant fields, U2's fixed
    # theta, and qutrit subspace rotations, bound inside the compiled sweep.
    a, b, c = fq.Parameter("a"), fq.Parameter("b"), fq.Parameter("c")
    qubits = fq.QuantumRegister(2)
    qutrit = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qubits, qutrit])
    program.add(ops.RX(a), qubits[0])
    program.add(ops.RY(b), qubits[1])
    program.add(ops.RZ(c), qubits[0])
    program.add(ops.Phase(a), qubits[1])
    program.add(ops.U(a, b, 0.3), qubits[0])
    program.add(ops.U1(b), qubits[1])
    program.add(ops.U2(a, c), qubits[0])
    program.add(ops.U3(c, a, b), qubits[1])
    program.add(ops.CPhase(b), (qubits[0], qubits[1]))
    program.add(ops.SubspaceRX(a, (0, 2)), qutrit[0])
    program.add(ops.SubspaceRY(b, (1, 2)), qutrit[0])
    program.add(ops.SubspaceRZ(c, (0, 1)), qutrit[0])
    values = np.random.default_rng(4).uniform(-np.pi, np.pi, (3, 5))
    bindings = {a: values[0], b: values[1], c: values[2]}

    swept = Simulator("SV").run_sweep(program, bindings, **_STATE_ONLY).result()
    reference = (
        Simulator("SV", runtime="numpy").run_sweep(program, bindings, **_STATE_ONLY)
    ).result()

    for result, expected in zip(swept, reference, strict=True):
        np.testing.assert_allclose(
            result.get_statevector(), expected.get_statevector(), rtol=0, atol=1e-12
        )
