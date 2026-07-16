import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.backends.engine_contract import _EngineConfig, _StateVectorResultRequest
from fatqat.simulator.parallel import _planned_workers


def test_backend_accepts_known_options():
    backend = SimulatorBackend("SV", 
        options={"max_workers": 1, "parallel_mode": "serial"}
    )

    assert backend._simulator.config.max_workers == 1
    assert backend._simulator.config.parallel_mode == "serial"


def test_backend_warns_and_ignores_unknown_options():
    with pytest.warns(UserWarning, match="ignored unsupported backend options"):
        backend = SimulatorBackend("SV", options={"gpu": True, "foo": 3})

    assert backend._simulator.config.max_workers is None
    assert backend._simulator.config.parallel_mode == "auto"


def test_serial_backend_option_runs_dynamic_program():
    p = fq.Program(1, 1)
    p.add(fq.ops.H, 0)
    p.add_measurement(0, 0)
    p.add(fq.ops.Reset, 0)

    counts = (
        SimulatorBackend("SV", options={"max_workers": 4, "parallel_mode": "serial"})
        .run(p, shots=16, seed=123)
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


def test_backend_rejects_non_dict_options():
    with pytest.raises(TypeError, match="dict or None"):
        SimulatorBackend("SV", options=object())


def test_backend_accepts_custom_implementation_map():
    from fatqat.implementation import default_matrix_implementation_map

    # Start from the defaults and override just one gate, to prove override
    # replaces one rule while the rest of the default map keeps working.
    m = default_matrix_implementation_map()
    m.add(fq.ops.X, np.eye(2, dtype=complex))  # override X with identity
    backend = SimulatorBackend("SV", implementation_map=m)

    p = fq.Program(2, 2)
    p.add(fq.ops.X, 0)  # overridden: identity, so qubit 0 stays |0>
    p.add(fq.ops.H, 1)  # still a default rule: H|0> is an equal superposition
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    counts = backend.run(p, shots=200, seed=0).result().get_counts()

    assert set(counts) <= {"00", "10"}  # c1 (H) varies; c0 (overridden X) is always 0
    assert counts.get("00", 0) + counts.get("10", 0) == 200


def test_backend_none_implementation_map_uses_defaults():
    backend = SimulatorBackend("SV", implementation_map=None)

    p = fq.Program(1, 1)
    p.add(fq.ops.X, 0)
    p.add_measurement(0, 0)
    counts = backend.run(p, shots=10, seed=0).result().get_counts()

    assert counts == {"1": 10}


def test_backend_copies_implementation_map_defensively():
    from fatqat.implementation import default_matrix_implementation_map

    m = default_matrix_implementation_map()
    backend = SimulatorBackend("SV", implementation_map=m)

    m.remove(fq.ops.X)  # mutate the caller's map after construction

    p = fq.Program(1, 1)
    p.add(fq.ops.X, 0)
    p.add_measurement(0, 0)
    counts = backend.run(p, shots=10, seed=0).result().get_counts()

    assert counts == {"1": 10}
