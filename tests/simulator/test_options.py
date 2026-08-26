import numpy as np
import pytest

import fatqat as fq
from fatqat._backends.engine_contract import (
    _SimulationConfig,
)
from fatqat.simulator._execution_contract import (
    _EngineCapabilities,
    _ExecutionPolicy,
    _PlanFacts,
)
from fatqat.simulator._execution_policy import (
    _materialization_policy,
    _process_child_policy,
    _resolve_execution_policy,
)
from fatqat.errors import BackendValidationError
from fatqat.simulator import Simulator


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


def test_kernel_threads_require_engine_support_before_execution():
    program = fq.Program(1, 1)
    program.measure(0, 0)

    with pytest.raises(BackendValidationError, match="threaded numerical kernels"):
        Simulator("unitary", runtime="numpy").run(
            program, simulation_config={"kernel_parallelism": "threads"}
        )


def test_materialization_failure_belongs_to_the_job(monkeypatch):
    backend = Simulator("SV", runtime="numpy")
    original = backend._engine.materialize_execution
    first = True

    def fail_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("materialization failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(backend._engine, "materialize_execution", fail_once)
    job = backend.run(fq.Program(1))

    with pytest.raises(RuntimeError, match="materialization failed"):
        job.result()

    assert backend.run(fq.Program(1)).result().get_statevector().tolist() == [
        1 + 0j,
        0j,
    ]


def _facts(execution_shape):
    return _PlanFacts(
        execution_shape=execution_shape,
        deferred_measurements=(),
        written_clbits=frozenset(),
        stochastic_final_state=False,
        has_measurement=False,
        has_reset=False,
        has_channel=False,
        has_condition=False,
    )


@pytest.mark.parametrize(
    (
        "simulation",
        "facts",
        "counts_requested",
        "state_requested",
        "capabilities",
        "compatible",
        "shots",
        "initial_occupied",
        "expected",
    ),
    [
        pytest.param(
            _SimulationConfig(),
            _facts("operator"),
            False,
            True,
            _EngineCapabilities(True, 8, False),
            False,
            1,
            None,
            ("none", "adaptive", None, False, False),
            id="operator-auto-adaptive-kernels",
        ),
        pytest.param(
            _SimulationConfig(),
            _facts("operator"),
            False,
            True,
            _EngineCapabilities(False, 1, False),
            False,
            1,
            None,
            ("none", "serial", 1, False, False),
            id="operator-auto-serial-without-thread-capability",
        ),
        pytest.param(
            _SimulationConfig(),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("threads", "serial", None, False, True),
            id="auto-compatible-compiled-shots",
        ),
        pytest.param(
            _SimulationConfig(max_workers=4),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            False,
            64,
            None,
            ("processes", "serial", 4, False, False),
            id="auto-large-incompatible-process-shots",
        ),
        pytest.param(
            _SimulationConfig(max_workers=4),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            False,
            8,
            None,
            ("serial", "adaptive", 4, False, False),
            id="auto-small-work-adaptive-kernels",
        ),
        pytest.param(
            _SimulationConfig(
                shot_parallelism="serial",
                kernel_parallelism="auto",
                max_workers=4,
            ),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("serial", "adaptive", 4, False, False),
            id="serial-shots-use-ordered-adaptive-replay",
        ),
        pytest.param(
            _SimulationConfig(
                shot_parallelism="serial",
                kernel_parallelism="threads",
                max_workers=99,
            ),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("serial", "threads", 8, False, False),
            id="serial-shots-forced-threaded-kernels",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="threads", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("threads", "serial", 8, False, True),
            id="explicit-shot-threads-have-concrete-capacity",
        ),
        pytest.param(
            _SimulationConfig(
                shot_parallelism="processes",
                kernel_parallelism="serial",
                fusion=True,
            ),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, True),
            False,
            64,
            None,
            ("processes", "serial", 6, True, False),
            id="explicit-process-shots-resolve-stable-capacity",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="serial", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("serial", "serial", 1, False, True),
            id="explicit-serial-uses-compatible-compiled-loop",
        ),
        pytest.param(
            _SimulationConfig(max_workers=1),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            None,
            ("serial", "serial", 1, False, True),
            id="auto-one-worker-uses-serial-compiled-loop",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="serial", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            64,
            frozenset(),
            ("serial", "serial", 1, False, False),
            id="initial-occupancy-disables-compiled-loop",
        ),
    ],
)
def test_execution_policy_decision_table(
    simulation,
    facts,
    counts_requested,
    state_requested,
    capabilities,
    compatible,
    shots,
    initial_occupied,
    expected,
    monkeypatch,
):
    monkeypatch.setattr(
        "fatqat.simulator._execution_policy.os.process_cpu_count",
        lambda: 6,
        raising=False,
    )
    policy = _resolve_execution_policy(
        simulation,
        facts=facts,
        counts_requested=counts_requested,
        state_requested=state_requested,
        capabilities=capabilities,
        compiled_multi_shot_compatible=compatible,
        shots=shots,
        initial_occupied=initial_occupied,
    )

    assert (
        policy.shot_strategy,
        policy.kernel_strategy,
        policy.worker_limit,
        policy.fusion,
        policy.use_compiled_multi_shot_kernel,
    ) == expected
    assert not (
        policy.shot_strategy in {"threads", "processes"}
        and policy.kernel_strategy == "threads"
    )
    assert not policy.use_compiled_multi_shot_kernel or (
        policy.kernel_strategy == "serial"
    )


@pytest.mark.parametrize(
    (
        "simulation",
        "facts",
        "counts_requested",
        "state_requested",
        "capabilities",
        "compatible",
        "shots",
        "plan_is_empty",
        "match",
    ),
    [
        pytest.param(
            _SimulationConfig(shot_parallelism="threads", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            True,
            1,
            False,
            "at least two shots",
            id="threaded-shots-need-two-shots",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="serial", kernel_parallelism="threads"),
            _facts("single_pass"),
            False,
            True,
            _EngineCapabilities(True, 1, False),
            False,
            1,
            False,
            "thread capacity of at least two",
            id="threaded-kernels-need-capacity-for-nonempty-work",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="threads", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 1, False),
            True,
            64,
            False,
            "thread capacity of at least two",
            id="threaded-shots-need-capacity",
        ),
        pytest.param(
            _SimulationConfig(
                shot_parallelism="processes", kernel_parallelism="serial"
            ),
            _facts("single_pass"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            False,
            64,
            False,
            "requires independent per-shot evolution",
            id="parallel-shots-need-per-shot-execution",
        ),
        pytest.param(
            _SimulationConfig(shot_parallelism="threads", kernel_parallelism="serial"),
            _facts("per_shot"),
            True,
            False,
            _EngineCapabilities(True, 8, False),
            False,
            64,
            False,
            "compiled multi-shot support from the selected engine",
            id="threaded-shots-need-compiled-compatibility",
        ),
        pytest.param(
            _SimulationConfig(
                shot_parallelism="processes", kernel_parallelism="serial"
            ),
            _facts("per_shot"),
            True,
            True,
            _EngineCapabilities(True, 8, False),
            False,
            64,
            False,
            "cannot shard the requested result fields",
            id="process-shots-need-shardable-results",
        ),
    ],
)
def test_execution_policy_rejects_inapplicable_requests(
    simulation,
    facts,
    counts_requested,
    state_requested,
    capabilities,
    compatible,
    shots,
    plan_is_empty,
    match,
):
    with pytest.raises(BackendValidationError, match=match):
        _resolve_execution_policy(
            simulation,
            facts=facts,
            counts_requested=counts_requested,
            state_requested=state_requested,
            capabilities=capabilities,
            compiled_multi_shot_compatible=compatible,
            shots=shots,
            initial_occupied=None,
            plan_is_empty=plan_is_empty,
        )


def test_process_policy_projections():
    local = _ExecutionPolicy(
        shot_strategy="serial",
        kernel_strategy="adaptive",
        worker_limit=None,
        fusion=False,
    )
    process = _ExecutionPolicy(
        shot_strategy="processes",
        kernel_strategy="serial",
        worker_limit=4,
        fusion=True,
    )

    assert _materialization_policy(local) == local
    assert _materialization_policy(process) == _ExecutionPolicy(
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=True,
    )
    assert _process_child_policy(process) == _ExecutionPolicy(
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=False,
    )


def test_run_rejects_non_dict_simulation_config():
    with pytest.raises(TypeError, match="dict or None"):
        Simulator("SV").run(fq.Program(1), simulation_config=object())


def test_runtime_is_keyword_only():
    with pytest.raises(TypeError, match="positional arguments"):
        Simulator("SV", "numba")


def test_backend_accepts_custom_implementation_map():
    from fatqat.implementation import default_matrix_implementation_map

    implementation_map = default_matrix_implementation_map()
    implementation_map.add(fq.ops.X, np.eye(2, dtype=complex))
    backend = Simulator("SV", implementation_map=implementation_map)

    program = fq.Program(2, 2)
    program.add(fq.ops.X, 0)
    program.add(fq.ops.H, 1)
    program.measure(0, 0)
    program.measure(1, 1)
    counts = (
        backend.run(program, shots=200, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )

    assert set(counts) <= {"00", "10"}
    assert counts.get("00", 0) + counts.get("10", 0) == 200


def test_backend_none_implementation_map_uses_defaults():
    backend = Simulator("SV", implementation_map=None)

    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.measure(0, 0)

    assert backend.run(
        program, shots=10, simulation_config={"seed": 0}
    ).result().get_counts() == {"1": 10}


def test_backend_copies_implementation_map_defensively():
    from fatqat.implementation import default_matrix_implementation_map

    implementation_map = default_matrix_implementation_map()
    backend = Simulator("SV", implementation_map=implementation_map)
    implementation_map.remove(fq.ops.X)

    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.measure(0, 0)
    assert backend.run(
        program, shots=10, simulation_config={"seed": 0}
    ).result().get_counts() == {"1": 10}
