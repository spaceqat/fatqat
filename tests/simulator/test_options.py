from dataclasses import replace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import LossStep, MeasurementStep, ResetStep
from fatqat.simulator._execution_contract import (
    _RunContext,
    _ResultRequest,
    _SimulationConfig,
    _TrajectoryCapabilities,
)
from fatqat.errors import BackendValidationError
from fatqat.simulator import EvolutionState, Simulator


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"shot_parallelism": "cluster"}, "unsupported shot_parallelism"),
        ({"shot_parallelism": []}, "unsupported shot_parallelism"),
        ({"kernel_parallelism": "processes"}, "unsupported kernel_parallelism"),
        ({"kernel_parallelism": []}, "unsupported kernel_parallelism"),
        (
            {"shot_parallelism": "threads", "kernel_parallelism": "threads"},
            "cannot both be explicitly parallel",
        ),
        (
            {"shot_parallelism": "processes", "kernel_parallelism": "threads"},
            "cannot both be explicitly parallel",
        ),
        ({"max_workers": True}, "positive int"),
        ({"max_workers": 0}, "positive int"),
        ({"max_workers": 1.5}, "positive int"),
        (
            {"shot_parallelism": "threads", "max_workers": 1},
            "max_workers=1 contradicts",
        ),
        (
            {"shot_parallelism": "processes", "max_workers": 1},
            "max_workers=1 contradicts",
        ),
        (
            {"kernel_parallelism": "threads", "max_workers": 1},
            "max_workers=1 contradicts",
        ),
        ({"fusion": None}, "fusion must be a bool"),
        ({"error_handling": "ignore"}, "unsupported error_handling"),
        ({"error_handling": []}, "unsupported error_handling"),
    ],
)
def test_public_execution_configuration_rejects_invalid_values(config, match):
    with pytest.raises(BackendValidationError, match=match):
        Simulator("SV").run(fq.Program(1), simulation_config=config)


@pytest.mark.parametrize(
    "config",
    [
        {"shot_parallelism": "serial", "kernel_parallelism": "threads"},
        {"shot_parallelism": "serial", "max_workers": 4},
        {"max_workers": 1},
    ],
)
def test_public_execution_configuration_accepts_supported_values(config):
    normalized = _SimulationConfig(**config)

    assert normalized.shot_parallelism in {"auto", "serial", "threads", "processes"}
    assert normalized.kernel_parallelism in {"auto", "serial", "threads"}


def test_public_execution_configuration_defaults():
    assert _SimulationConfig() == _SimulationConfig(
        seed=None,
        shot_parallelism="auto",
        kernel_parallelism="auto",
        max_workers=None,
        fusion=False,
    )


@pytest.mark.parametrize(
    "request_kind, config, match",
    [
        ("run", {"kernel_parallelism": "threads"}, "threaded numerical kernels"),
        ("run", {"fusion": True}, "fusion=True is not supported"),
        ("run", {"shot_parallelism": "processes"}, "processes.*not supported"),
        ("sweep", {"kernel_parallelism": "threads"}, "threaded numerical kernels"),
        (
            "expectation",
            {"kernel_parallelism": "threads"},
            "threaded numerical kernels",
        ),
    ],
)
def test_engine_support_validates_execution_controls(
    monkeypatch, request_kind, config, match
):
    backend = Simulator("SV", runtime="numpy")
    monkeypatch.setattr(
        backend._engine,
        "_capabilities",
        replace(backend._engine.capabilities, supports_process_shots=False),
    )
    program = fq.Program(1)
    theta = fq.Parameter("theta")
    program.add(ops.RX(theta if request_kind == "sweep" else 0.1), 0)

    with pytest.raises(BackendValidationError, match=match):
        if request_kind == "sweep":
            backend.run_sweep(program, {theta: [0.1, 0.2]}, simulation_config=config)
        elif request_kind == "expectation":
            fq.Estimator(backend).run(
                program, fq.Observable([("Z", 1.0)]), simulation_config=config
            )
        else:
            backend.run(program, simulation_config=config)


@pytest.mark.parametrize("mode", [None, "raise", "capture"])
@pytest.mark.parametrize(
    "request_kind", ["run", "sweep", "expectation", "sampled_expectation"]
)
def test_execution_error_handling(monkeypatch, mode, request_kind):
    backend = Simulator("SV", runtime="numpy")
    error = RuntimeError("materialization failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(backend._engine, "prepare_payload", fail)
    config = {} if mode is None else {"error_handling": mode}
    program = fq.Program(1)
    theta = fq.Parameter("theta")
    program.add(ops.RX(theta if request_kind == "sweep" else 0.1), 0)

    def run():
        if request_kind == "sweep":
            return backend.run_sweep(
                program, {theta: [0.1, 0.2]}, simulation_config=config
            )
        if request_kind in {"expectation", "sampled_expectation"}:
            return fq.Estimator(backend).run(
                program,
                fq.Observable([("Z", 1.0)]),
                shots=8 if request_kind == "sampled_expectation" else 0,
                simulation_config=config,
            )
        return backend.run(program, simulation_config=config)

    if mode == "capture":
        job = run()
        with pytest.raises(RuntimeError) as caught:
            job.result()
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
    assert caught.value is error


@pytest.mark.parametrize("request_kind", ["run", "sweep", "expectation", "constant"])
def test_capture_returns_engine_planning_errors_in_job(request_kind):
    backend = Simulator("SV", runtime="numpy")
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RX(theta if request_kind == "sweep" else 0.1), 0)
    config = {"error_handling": "capture", "shot_parallelism": "processes"}

    def run():
        if request_kind == "sweep":
            return backend.run_sweep(program, {theta: [0.1]}, simulation_config=config)
        if request_kind in {"expectation", "constant"}:
            return fq.Estimator(backend).run(
                program,
                fq.Observable([("I" if request_kind == "constant" else "Z", 1.0)]),
                simulation_config=config,
            )
        return backend.run(program, simulation_config=config)

    job = run()
    with pytest.raises(BackendValidationError, match="requires independent per-shot"):
        job.result()


@pytest.mark.parametrize("operation", ["measurement", "condition"])
@pytest.mark.parametrize("sweep", [False, True])
def test_engine_classical_support_controls_early_validation(
    monkeypatch, operation, sweep
):
    backend = Simulator("SV", runtime="numpy")
    monkeypatch.setattr(
        backend._engine,
        "_capabilities",
        replace(backend._engine.capabilities, trajectory=_TrajectoryCapabilities()),
    )
    program = fq.Program(1, 1)
    theta = fq.Parameter("theta")
    if sweep:
        program.add(ops.RX(theta), 0)
    if operation == "measurement":
        program.measure(0, 0)
    else:
        program.add(ops.X, 0, condition=(program.classical_registers[0][0], 0))

    with pytest.raises(BackendValidationError, match="classical register"):
        if sweep:
            backend.run_sweep(program, {theta: [0.1, 0.2]})
        else:
            backend.run(program)


@pytest.mark.parametrize("operation", ["reset", "channel"])
def test_engine_quantum_support_controls_early_validation(monkeypatch, operation):
    noise = fq.NoiseModel()
    if operation == "channel":
        noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.X)
    backend = Simulator("SV", runtime="numpy", noise=noise)
    quantum = backend._engine.capabilities.quantum
    monkeypatch.setattr(
        backend._engine,
        "_capabilities",
        replace(
            backend._engine.capabilities,
            quantum=replace(quantum, supports_nonunitary=False),
        ),
    )
    program = fq.Program(1)
    program.add(ops.Reset if operation == "reset" else ops.X, 0)

    with pytest.raises(BackendValidationError, match="cannot execute"):
        backend.run(program)


@pytest.mark.parametrize(
    "shots, first_step, expected_workflow",
    [
        pytest.param(1, ResetStep((0,)), "python", id="single-trajectory"),
        pytest.param(
            64, ResetStep((0,)), "numba_trajectories", id="repeated-trajectories"
        ),
        pytest.param(
            64, LossStep((0,), p=0.1), "python", id="unsupported-step-fallback"
        ),
    ],
)
def test_numba_trajectory_workflow_selection(shots, first_step, expected_workflow):
    from fatqat.simulator._engine.engines.numba_engine import NumbaSVEngine

    # Numerical results alone cannot detect accidentally disabling compilation.
    policy = NumbaSVEngine()._select_policy(
        (first_step, MeasurementStep((0,), (0,))),
        context=_RunContext(
            request=_ResultRequest(counts=True, final_state=False),
            system_dims=(2,),
            n_clbits=1,
            shots=shots,
            seed=None,
        ),
        simulation_config=_SimulationConfig(max_workers=1),
        initial_state=EvolutionState(),
    )

    assert policy.workflow == expected_workflow


def test_process_worker_capacity_follows_the_cpu_affinity_mask(monkeypatch):
    from fatqat.simulator._engine.engines.numpy_engine import NumpySVEngine

    # Before Python 3.13, cpu_count ignores process affinity restrictions.
    module = "fatqat.simulator._engine._execution_policy.os"
    monkeypatch.delattr(f"{module}.process_cpu_count", raising=False)
    monkeypatch.setattr(f"{module}.cpu_count", lambda: 192)
    monkeypatch.setattr(
        f"{module}.sched_getaffinity", lambda pid: {4, 5, 6}, raising=False
    )
    policy = NumpySVEngine()._select_policy(
        (ResetStep((0,)), MeasurementStep((0,), (0,))),
        context=_RunContext(
            request=_ResultRequest(counts=True, final_state=False),
            system_dims=(2,),
            n_clbits=1,
            shots=64,
            seed=None,
        ),
        simulation_config=_SimulationConfig(
            shot_parallelism="processes", kernel_parallelism="serial"
        ),
        initial_state=EvolutionState(),
    )

    assert policy.worker_limit == 3


@pytest.fixture(name="dynamic_program")
def fixture_dynamic_program():
    program = fq.Program(1, 1)
    program.measure(0, 0)
    program.add(ops.X, 0)
    return program


@pytest.fixture(name="terminal_measurement_program")
def fixture_terminal_measurement_program():
    program = fq.Program(1, 1)
    program.measure(0, 0)
    return program


@pytest.mark.parametrize(
    "runtime, program_fixture, config, shots, result_config, match",
    [
        pytest.param(
            "numba",
            "dynamic_program",
            {"shot_parallelism": "threads"},
            1,
            {"counts": True},
            "at least two shots",
            id="threaded-shots-need-two-shots",
        ),
        pytest.param(
            "numpy",
            "terminal_measurement_program",
            {"shot_parallelism": "processes"},
            64,
            {"counts": True},
            "requires independent per-shot evolution",
            id="parallel-shots-need-per-shot-execution",
        ),
        pytest.param(
            "numpy",
            "dynamic_program",
            {"shot_parallelism": "threads"},
            64,
            {"counts": True},
            "not supported by the Python workflow",
            id="python-workflow-rejects-threaded-shots",
        ),
        pytest.param(
            "numpy",
            "dynamic_program",
            {"shot_parallelism": "processes"},
            64,
            {"counts": False, "final_state": False},
            "cannot shard the requested result fields",
            id="parallel-shots-need-results-to-collect",
        ),
    ],
)
def test_run_rejects_inapplicable_shot_parallelism(
    runtime, program_fixture, config, shots, result_config, match, request
):
    with pytest.raises(BackendValidationError, match=match):
        Simulator("SV", runtime=runtime).run(
            request.getfixturevalue(program_fixture),
            shots=shots,
            simulation_config=config,
            result_config=result_config,
        ).result()


@pytest.mark.parametrize("axis", ["kernel_parallelism", "shot_parallelism"])
def test_run_rejects_threading_without_worker_capacity(
    monkeypatch, dynamic_program, axis
):
    backend = Simulator("SV", runtime="numba")
    monkeypatch.setattr(
        backend._engine,
        "_capabilities",
        replace(
            backend._engine.capabilities,
            kernels=replace(backend._engine.capabilities.kernels, thread_capacity=1),
        ),
    )
    with pytest.raises(BackendValidationError, match="thread capacity of at least two"):
        backend.run(
            dynamic_program, shots=64, simulation_config={axis: "threads"}
        ).result()


def test_run_rejects_non_dict_simulation_config():
    with pytest.raises(TypeError, match="dict or None"):
        Simulator("SV").run(fq.Program(1), simulation_config=object())


def test_runtime_is_keyword_only():
    with pytest.raises(TypeError, match="positional arguments"):
        Simulator("SV", "numba")


def test_backend_accepts_custom_implementation_map():
    from fatqat.implementation import default_matrix_implementation_map

    implementation_map = default_matrix_implementation_map()
    implementation_map.add(ops.X, np.eye(2, dtype=complex))
    backend = Simulator("SV", implementation_map=implementation_map)

    program = fq.Program(2, 2)
    program.add(ops.X, 0)
    program.add(ops.H, 1)
    program.measure(0, 0)
    program.measure(1, 1)
    counts = (
        backend.run(program, shots=200, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )

    assert set(counts) <= {"00", "01"}
    assert counts.get("00", 0) + counts.get("01", 0) == 200


def test_backend_none_implementation_map_uses_defaults():
    backend = Simulator("SV", implementation_map=None)

    program = fq.Program(1, 1)
    program.add(ops.X, 0)
    program.measure(0, 0)

    assert backend.run(
        program, shots=10, simulation_config={"seed": 0}
    ).result().get_counts() == {"1": 10}


def test_backend_copies_implementation_map_defensively():
    from fatqat.implementation import default_matrix_implementation_map

    implementation_map = default_matrix_implementation_map()
    backend = Simulator("SV", implementation_map=implementation_map)
    implementation_map.remove(ops.X)

    program = fq.Program(1, 1)
    program.add(ops.X, 0)
    program.measure(0, 0)
    assert backend.run(
        program, shots=10, simulation_config={"seed": 0}
    ).result().get_counts() == {"1": 10}
