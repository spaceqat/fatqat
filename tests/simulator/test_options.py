import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import Simulator
from fatqat._backends.engine_contract import _EngineConfig, _StateVectorResultRequest
from fatqat.simulator._engine.parallel import _planned_workers


def test_run_accepts_known_simulation_config_without_mutating_backend():
    backend = Simulator("SV")
    backend.run(
        fq.Program(1),
        simulation_config={"max_workers": 1, "parallel_mode": "serial"},
    )

    assert backend._engine.config.max_workers is None
    assert backend._engine.config.parallel_mode == "auto"


def test_run_rejects_unknown_simulation_config():
    with pytest.raises(
        fq.errors.BackendValidationError, match="does not support simulation_config"
    ):
        Simulator("SV").run(fq.Program(1), simulation_config={"gpu": True})


def test_serial_simulation_config_runs_dynamic_program():
    p = fq.Program(1, 1)
    p.add(fq.ops.H, 0)
    p.measure(0, 0)
    p.add(fq.ops.Reset, 0)

    counts = (
        Simulator("SV")
        .run(
            p,
            shots=16,
            simulation_config={
                "seed": 123,
                "max_workers": 4,
                "parallel_mode": "serial",
            },
        )
        .result()
        .get_counts()
    )

    assert sum(counts.values()) == 16


def test_planned_workers_disables_parallel_serial_backend():
    workers = _planned_workers(
        _EngineConfig(max_workers=4, parallel_mode="serial"),
        _StateVectorResultRequest(counts=True, statevector=False),
        n_iters=16,
    )

    assert workers is None


def test_planned_workers_clamps_explicit_workers_to_iterations():
    workers = _planned_workers(
        _EngineConfig(max_workers=8, parallel_mode="multiprocessing"),
        _StateVectorResultRequest(counts=True, statevector=False),
        n_iters=3,
    )

    assert workers == 3


def test_run_rejects_non_dict_simulation_config():
    with pytest.raises(TypeError, match="dict or None"):
        Simulator("SV").run(fq.Program(1), simulation_config=object())


def test_runtime_is_keyword_only():
    with pytest.raises(TypeError, match="positional arguments"):
        Simulator("SV", "numba")


def test_backend_accepts_custom_implementation_map():
    from fatqat.implementation import default_matrix_implementation_map

    # Start from the defaults and override just one gate, to prove override
    # replaces one rule while the rest of the default map keeps working.
    m = default_matrix_implementation_map()
    m.add(fq.ops.X, np.eye(2, dtype=complex))  # override X with identity
    backend = Simulator("SV", implementation_map=m)

    p = fq.Program(2, 2)
    p.add(fq.ops.X, 0)  # overridden: identity, so qubit 0 stays |0>
    p.add(fq.ops.H, 1)  # still a default rule: H|0> is an equal superposition
    p.measure(0, 0)
    p.measure(1, 1)
    counts = (
        backend.run(p, shots=200, simulation_config={"seed": 0}).result().get_counts()
    )

    assert set(counts) <= {"00", "10"}  # c1 (H) varies; c0 (overridden X) is always 0
    assert counts.get("00", 0) + counts.get("10", 0) == 200


def test_backend_none_implementation_map_uses_defaults():
    backend = Simulator("SV", implementation_map=None)

    p = fq.Program(1, 1)
    p.add(fq.ops.X, 0)
    p.measure(0, 0)
    counts = (
        backend.run(p, shots=10, simulation_config={"seed": 0}).result().get_counts()
    )

    assert counts == {"1": 10}


def test_backend_copies_implementation_map_defensively():
    from fatqat.implementation import default_matrix_implementation_map

    m = default_matrix_implementation_map()
    backend = Simulator("SV", implementation_map=m)

    m.remove(fq.ops.X)  # mutate the caller's map after construction

    p = fq.Program(1, 1)
    p.add(fq.ops.X, 0)
    p.measure(0, 0)
    counts = (
        backend.run(p, shots=10, simulation_config={"seed": 0}).result().get_counts()
    )

    assert counts == {"1": 10}
